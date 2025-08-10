# group_splitter_app.py
# Run locally:
#   pip install streamlit pandas
#   streamlit run group_splitter_app.py

import sqlite3
from contextlib import closing
from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

DB_PATH = "group_splitter.db"
CURRENCY = "AUD"

# ---------- Utilities ----------


def d2c(amount_str: str) -> int:
    """Decimal string -> cents (int). Safer than float."""
    d = Decimal(amount_str).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int((d * 100))


def c2d(cents: int) -> Decimal:
    return (Decimal(cents) / Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def fmt_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{CURRENCY} {c2d(cents)}"


# ---------- Database ----------


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(get_conn()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE(group_id, name),
                FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                tx_date TEXT NOT NULL,
                description TEXT,
                amount_cents INTEGER NOT NULL,
                payer_id INTEGER NOT NULL,
                FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE,
                FOREIGN KEY(payer_id) REFERENCES members(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expense_participants (
                expense_id INTEGER NOT NULL,
                member_id INTEGER NOT NULL,
                PRIMARY KEY(expense_id, member_id),
                FOREIGN KEY(expense_id) REFERENCES expenses(id) ON DELETE CASCADE,
                FOREIGN KEY(member_id) REFERENCES members(id) ON DELETE CASCADE
            );
            """
        )


def create_group(name: str) -> int:
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO groups(name) VALUES (?)", (name.strip(),)
        )
        if cur.lastrowid is None:
            # fetch existing id
            cur = conn.execute("SELECT id FROM groups WHERE name=?", (name.strip(),))
        return cur.lastrowid or cur.fetchone()[0]


def list_groups() -> pd.DataFrame:
    with closing(get_conn()) as conn:
        df = pd.read_sql_query("SELECT id, name FROM groups ORDER BY name", conn)
    return df


def add_member(group_id: int, name: str):
    with closing(get_conn()) as conn, conn:
        conn.execute(
            "INSERT OR IGNORE INTO members(group_id, name) VALUES (?, ?)",
            (group_id, name.strip()),
        )


def list_members(group_id: int) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        df = pd.read_sql_query(
            "SELECT id, name FROM members WHERE group_id=? ORDER BY name",
            conn,
            params=(group_id,),
        )
    return df


def rename_member(member_id: int, new_name: str):
    with closing(get_conn()) as conn, conn:
        conn.execute(
            "UPDATE members SET name=? WHERE id=?", (new_name.strip(), member_id)
        )


def delete_group(group_id: int):
    with closing(get_conn()) as conn, conn:
        # Ensure cascades; also do manual cleanup as a fallback
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            DELETE FROM expense_participants
            WHERE expense_id IN (SELECT id FROM expenses WHERE group_id=?)
        """,
            (group_id,),
        )
        conn.execute("DELETE FROM expenses WHERE group_id=?", (group_id,))
        conn.execute("DELETE FROM members WHERE group_id=?", (group_id,))
        conn.execute("DELETE FROM groups WHERE id=?", (group_id,))


def add_expense(
    group_id: int,
    tx_date: date,
    description: str,
    amount_cents: int,
    payer_id: int,
    participant_ids: List[int],
):
    if not participant_ids:
        raise ValueError("At least one participant required")
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO expenses(group_id, tx_date, description, amount_cents, payer_id) VALUES (?, ?, ?, ?, ?)",
            (
                group_id,
                tx_date.isoformat(),
                description.strip(),
                amount_cents,
                payer_id,
            ),
        )
        expense_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO expense_participants(expense_id, member_id) VALUES (?, ?)",
            [(expense_id, mid) for mid in participant_ids],
        )


def list_expenses(group_id: int) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        df = pd.read_sql_query(
            """
            SELECT e.id, e.tx_date AS date, e.description, e.amount_cents, m.name AS payer
            FROM expenses e
            JOIN members m ON m.id = e.payer_id
            WHERE e.group_id=?
            ORDER BY e.tx_date DESC, e.id DESC
            """,
            conn,
            params=(group_id,),
        )
    df["amount"] = df["amount_cents"].apply(fmt_money)
    return df[["id", "date", "description", "payer", "amount"]]


def expense_participants(expense_id: int) -> List[int]:
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT member_id FROM expense_participants WHERE expense_id=?",
            (expense_id,),
        ).fetchall()
    return [r[0] for r in rows]


# ---------- Balances & Settlement ----------


def compute_net_balances(group_id: int) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Returns (net_cents_by_member, names)
    net>0: others owe them; net<0: they owe others.
    """
    members = list_members(group_id)  # columns: id, name
    # robust id->name map
    id_to_name = dict(zip(members["id"].astype(int), members["name"].astype(str)))
    net = {mid: 0 for mid in id_to_name.keys()}

    with closing(get_conn()) as conn:
        expenses = conn.execute(
            "SELECT id, amount_cents, payer_id FROM expenses WHERE group_id=?",
            (group_id,),
        ).fetchall()

    for exp_id, amount_cents, payer_id in expenses:
        parts = expense_participants(exp_id)
        if not parts:
            continue
        # Equal split in cents, distribute remainder fairly
        share = amount_cents // len(parts)
        # Payer paid full amount
        net[payer_id] += amount_cents
        # Each participant owes their share
        for mid in parts:
            net[mid] -= share
        # Adjust rounding remainder to keep totals consistent
        remainder = amount_cents - share * len(parts)
        for i in range(abs(remainder)):
            # distribute the leftover 1c to first participants
            target = parts[i % len(parts)]
            net[target] -= 1 if remainder > 0 else 0

    return net, id_to_name


def optimize_settlements(net: Dict[int, int]) -> List[Tuple[int, int, int]]:
    """
    Greedy minimize transfers: returns list of (debtor_id, creditor_id, amount_cents).
    """
    debtors = [(mid, -amt) for mid, amt in net.items() if amt < 0]  # amounts they owe
    creditors = [(mid, amt) for mid, amt in net.items() if amt > 0]
    debtors.sort(key=lambda x: x[1], reverse=True)
    creditors.sort(key=lambda x: x[1], reverse=True)

    i = j = 0
    transfers = []
    while i < len(debtors) and j < len(creditors):
        d_id, d_amt = debtors[i]
        c_id, c_amt = creditors[j]
        pay = min(d_amt, c_amt)
        if pay > 0:
            transfers.append((d_id, c_id, pay))
        d_amt -= pay
        c_amt -= pay
        if d_amt == 0:
            i += 1
        else:
            debtors[i] = (d_id, d_amt)
        if c_amt == 0:
            j += 1
        else:
            creditors[j] = (c_id, c_amt)
    return transfers


def list_expenses_with_participants(group_id: int) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        df = pd.read_sql_query(
            """
            SELECT e.id, e.tx_date AS date, e.description, e.amount_cents,
                   e.payer_id, m.name AS payer
            FROM expenses e
            JOIN members m ON m.id = e.payer_id
            WHERE e.group_id=?
            ORDER BY e.tx_date DESC, e.id DESC
            """,
            conn,
            params=(group_id,),
        )
        if not df.empty:
            ids = tuple(int(x) for x in df["id"].tolist())
            q_marks = ",".join(["?"] * len(ids))
            part_map = {}
            rows = conn.execute(
                f"""SELECT ep.expense_id, mm.name
                    FROM expense_participants ep
                    JOIN members mm ON mm.id = ep.member_id
                    WHERE ep.expense_id IN ({q_marks})""",
                ids,
            ).fetchall()
            for exp_id, name in rows:
                part_map.setdefault(exp_id, []).append(name)
            df["participants"] = df["id"].map(lambda x: ", ".join(part_map.get(x, [])))
    df["amount"] = df["amount_cents"].apply(fmt_money)
    return df[
        [
            "id",
            "date",
            "description",
            "payer",
            "participants",
            "amount",
            "amount_cents",
            "payer_id",
        ]
    ]


def update_expense(
    expense_id: int,
    tx_date: date,
    description: str,
    amount_cents: int,
    payer_id: int,
    participant_ids: List[int],
):
    with closing(get_conn()) as conn, conn:
        conn.execute(
            "UPDATE expenses SET tx_date=?, description=?, amount_cents=?, payer_id=? WHERE id=?",
            (
                tx_date.isoformat(),
                description.strip(),
                amount_cents,
                payer_id,
                expense_id,
            ),
        )
        conn.execute(
            "DELETE FROM expense_participants WHERE expense_id=?", (expense_id,)
        )
        conn.executemany(
            "INSERT INTO expense_participants(expense_id, member_id) VALUES (?, ?)",
            [(expense_id, mid) for mid in participant_ids],
        )


def delete_expense(expense_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute(
            "DELETE FROM expense_participants WHERE expense_id=?", (expense_id,)
        )
        conn.execute("DELETE FROM expenses WHERE id=?", (expense_id,))


def _rerun():
    # Works on both new & old Streamlit
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------- UI ----------


def main():
    st.set_page_config(page_title="Group Splitter", page_icon="💸", layout="wide")
    st.title("💸 Group Splitter (local & private)")
    st.caption(
        "SQLite storage • Equal splits • Smart settlement • Currency: " + CURRENCY
    )

    init_db()

    with st.sidebar:
        st.header("Group")
        groups_df = list_groups()
        group_names = groups_df["name"].tolist() if not groups_df.empty else []
        new_group = st.text_input(
            "Create or select a group", placeholder="e.g., Ski Trip"
        )
        selected_group = None
        if st.button("Use group"):
            if new_group.strip():
                gid = create_group(new_group.strip())
                st.session_state["group_id"] = gid
                st.session_state["group_name"] = new_group.strip()
        if group_names:
            picked = st.selectbox(
                "Existing groups",
                group_names,
                index=(
                    0
                    if "group_name" not in st.session_state
                    else max(
                        (
                            group_names.index(st.session_state["group_name"])
                            if st.session_state["group_name"] in group_names
                            else 0
                        ),
                        0,
                    )
                ),
            )
            if st.button("Load selected"):
                gid = int(groups_df.loc[groups_df["name"] == picked, "id"].iloc[0])
                st.session_state["group_id"] = gid
                st.session_state["group_name"] = picked

        if "group_id" in st.session_state:
            st.divider()
            st.subheader("Danger zone")
            if st.button("Delete this group", key="delete_group_btn"):
                st.session_state["confirm_delete_group"] = True

            if st.session_state.get("confirm_delete_group"):
                st.warning(
                    f"Delete group **{st.session_state['group_name']}** and ALL its data?"
                )
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete", key="confirm_delete_yes"):
                    delete_group(st.session_state["group_id"])
                    st.success("Group deleted.")
                    st.session_state.pop("group_id", None)
                    st.session_state.pop("group_name", None)
                    # refresh
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                if c2.button("Cancel", key="confirm_delete_no"):
                    st.session_state["confirm_delete_group"] = False

    if "group_id" not in st.session_state:
        st.info("Create or select a group in the sidebar to begin.")
        return

    group_id = st.session_state["group_id"]
    st.subheader(f"Group: {st.session_state['group_name']}")

    tabs = st.tabs(["Members", "Add expense", "Expenses", "Balances & Settlement"])

    # --- Members tab ---
    with tabs[0]:
        st.markdown("### Members")

        # Add new member
        name = st.text_input("Add member", placeholder="e.g., Alice")
        if st.button("Add member") and name.strip():
            try:
                add_member(group_id, name)
                st.success(f"Added member: {name}")
                _rerun()  # uses st.rerun() or experimental fallback
            except sqlite3.IntegrityError:
                st.error("That name already exists in this group.")

        # Rename existing members (no deletion allowed)
        members_df = list_members(group_id)
        if members_df.empty:
            st.info("No members yet.")
        else:
            for _, r in members_df.iterrows():
                mid = int(r["id"])
                mname = str(r["name"])
                with st.expander(mname):
                    new_name = st.text_input("Name", value=mname, key=f"mname_{mid}")
                    if st.button("Save", key=f"msave_{mid}"):
                        try:
                            if new_name.strip() and new_name.strip() != mname:
                                rename_member(mid, new_name)
                                st.success("Member updated.")
                                _rerun()
                        except sqlite3.IntegrityError:
                            st.error(
                                "A member with that name already exists in this group."
                            )

        st.caption("Note: Members cannot be deleted; rename if needed.")

    # --- Add expense tab ---
    with tabs[1]:
        st.markdown("### Add expense")
        members_df = list_members(group_id)
        if members_df.empty:
            st.warning("Add at least one member first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                tx_date = st.date_input("Date", value=date.today())
                description = st.text_input(
                    "Description", placeholder="e.g., Groceries"
                )
                amount_str = st.text_input("Amount (e.g., 123.45)")
            with col2:
                payer = st.selectbox("Payer", members_df["name"].tolist())
                default_parts = members_df["name"].tolist()
                participants = st.multiselect(
                    "Participants (equal split)", default_parts, default=default_parts
                )

            if st.button("Save expense"):
                try:
                    amt_cents = d2c(amount_str)
                    payer_id = int(
                        members_df.loc[members_df["name"] == payer, "id"].iloc[0]
                    )
                    participant_ids = [
                        int(members_df.loc[members_df["name"] == n, "id"].iloc[0])
                        for n in participants
                    ]
                    add_expense(
                        group_id,
                        tx_date,
                        description,
                        amt_cents,
                        payer_id,
                        participant_ids,
                    )
                    st.success("Expense saved")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Expenses tab ---
    with tabs[2]:
        st.markdown("### Expenses")
        exp_df = list_expenses_with_participants(group_id)
        if exp_df.empty:
            st.info("No expenses yet.")
        else:
            members_df = list_members(group_id)
            member_names = members_df["name"].astype(str).tolist()
            id_by_name = dict(
                zip(members_df["name"].astype(str), members_df["id"].astype(int))
            )

            for _, row in exp_df.iterrows():
                eid = int(row["id"])
                with st.expander(
                    f"{row['date']} · {row['description']} · {row['payer']} · {row['amount']}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_date = st.date_input(
                            "Date",
                            value=pd.to_datetime(row["date"]).date(),
                            key=f"edate_{eid}",
                        )
                        new_desc = st.text_input(
                            "Description", value=row["description"], key=f"edesc_{eid}"
                        )
                        new_amount = st.text_input(
                            "Amount (e.g., 12.34)",
                            value=str(c2d(int(row["amount_cents"]))),
                            key=f"eamt_{eid}",
                        )
                    with col2:
                        new_payer = st.selectbox(
                            "Payer",
                            member_names,
                            index=(
                                member_names.index(row["payer"])
                                if row["payer"] in member_names
                                else 0
                            ),
                            key=f"epayer_{eid}",
                        )
                        current_parts = [
                            p.strip()
                            for p in (row["participants"] or "").split(",")
                            if p.strip()
                        ]
                        default_parts = current_parts if current_parts else member_names
                        new_parts = st.multiselect(
                            "Participants",
                            member_names,
                            default=default_parts,
                            key=f"eparts_{eid}",
                        )

                    b1, b2 = st.columns([1, 1])
                    if b1.button("Update", key=f"update_{eid}"):
                        try:
                            amt_cents = d2c(new_amount)
                            payer_id = id_by_name[new_payer]
                            part_ids = [id_by_name[n] for n in new_parts]
                            update_expense(
                                eid, new_date, new_desc, amt_cents, payer_id, part_ids
                            )
                            st.success("Updated.")
                            _rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                    if b2.button("Delete", key=f"delete_{eid}"):
                        delete_expense(eid)
                        st.warning("Deleted.")
                        _rerun()

    # --- Balances & Settlement tab ---
    with tabs[3]:
        st.markdown("### Balances & Settlement")
        net, id_to_name = compute_net_balances(group_id)
        if not id_to_name:
            st.info("Add members and expenses to see balances.")
        else:
            # Net table
            net_rows = [
                {
                    "Member": id_to_name.get(mid, f"Member {mid}"),
                    "Net": fmt_money(amt),
                    "Status": (
                        "is owed" if amt > 0 else ("owes" if amt < 0 else "settled")
                    ),
                }
                for mid, amt in sorted(
                    net.items(), key=lambda kv: str(id_to_name.get(kv[0], "")).lower()
                )
            ]
            st.markdown("**Net positions** (positive means others owe them):")
            st.dataframe(
                pd.DataFrame(net_rows), hide_index=True, use_container_width=True
            )

            transfers = optimize_settlements(net)
            if transfers:
                st.markdown("**Suggested minimal settlements:**")
                t_rows = [
                    {"From": id_to_name[d], "To": id_to_name[c], "Pay": fmt_money(a)}
                    for d, c, a in transfers
                ]
                st.dataframe(
                    pd.DataFrame(t_rows), hide_index=True, use_container_width=True
                )
            else:
                st.success("All settled! 🎉")


if __name__ == "__main__":
    main()
