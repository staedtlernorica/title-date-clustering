import streamlit as st
import pandas as pd
import copy
import time

# --- Sample video titles ---
titles = [
    "william mckinly",
    "mutiny on the bounty",
    "history of cats",
    "nazi occultism",
    "benedict arnold"
]

# --- Session state initialization ---
if "tagged_data" not in st.session_state:
    st.session_state.tagged_data = {title: "" for title in titles}

if "mass_checkboxes" not in st.session_state:
    st.session_state.mass_checkboxes = {title: False for title in titles}

if "dropdown_select" not in st.session_state:
    st.session_state.dropdown_select = []

if "history" not in st.session_state:
    st.session_state.history = [copy.deepcopy(st.session_state.tagged_data)]

if "future" not in st.session_state:
    st.session_state.future = []

if "recently_modified" not in st.session_state:
    st.session_state.recently_modified = set()

if "recently_added_tags" not in st.session_state:
    st.session_state.recently_added_tags = {}

if "select_all_clicked" not in st.session_state:
    st.session_state.select_all_clicked = False

if "clear_all_clicked" not in st.session_state:
    st.session_state.clear_all_clicked = False

# --- Handle Select All / Clear BEFORE widgets render ---
if st.session_state.select_all_clicked:
    for title in titles:
        st.session_state.mass_checkboxes[title] = True
    st.session_state.dropdown_select = titles[:]
    st.session_state.select_all_clicked = False
    st.rerun()

if st.session_state.clear_all_clicked:
    for title in titles:
        st.session_state.mass_checkboxes[title] = False
    st.session_state.dropdown_select = []
    st.session_state.clear_all_clicked = False
    st.rerun()

# --- Title ---
st.title("🎬 Mass Tagging Interface")

# --- Select All / Clear Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Select All"):
        st.session_state.select_all_clicked = True
        st.rerun()
with col2:
    if st.button("❌ Clear Selection"):
        st.session_state.clear_all_clicked = True
        st.rerun()

# --- Dropdown Selection ---
dropdown_selected = st.multiselect(
    "🔽 Select videos (dropdown):",
    titles,
    default=st.session_state.dropdown_select,
    key="dropdown_select"
)

# --- Sync checkboxes from dropdown ---
for title in titles:
    st.session_state.mass_checkboxes[title] = title in dropdown_selected

# --- Checkbox list ---
st.markdown("### ✅ Or check videos below:")
for title in titles:
    st.session_state.mass_checkboxes[title] = st.checkbox(
        label=title,
        value=st.session_state.mass_checkboxes[title],
        key=f"checkbox_{title}"
    )

# --- Tagging Form ---
with st.form("tag_form"):
    st.markdown("### 🏷️ Enter tags for selected videos (comma-separated):")
    mass_tag_input = st.text_input("Tags:")
    submitted = st.form_submit_button("Apply Tags to Selected")

    if submitted:
        selected_titles = [title for title, checked in st.session_state.mass_checkboxes.items() if checked]
        new_tags = [tag.strip() for tag in mass_tag_input.split(",") if tag.strip()]

        if not selected_titles:
            st.warning("No videos selected.")
        elif not new_tags:
            st.warning("No tags entered.")
        else:
            # Save history
            st.session_state.history.append(copy.deepcopy(st.session_state.tagged_data))
            st.session_state.future.clear()
            st.session_state.recently_modified = set()
            st.session_state.recently_added_tags = {}

            for title in selected_titles:
                existing = st.session_state.tagged_data[title]
                existing_tags = set(t.strip() for t in existing.split(",") if t.strip())
                new_tags_set = set(new_tags)
                added_tags = new_tags_set - existing_tags

                # Update tags
                all_tags = existing_tags.union(new_tags_set)
                st.session_state.tagged_data[title] = ", ".join(sorted(all_tags))

                if added_tags:
                    st.session_state.recently_added_tags[title] = list(added_tags)

                st.session_state.recently_modified.add(title)

            # Flag to trigger highlight view
            st.experimental_set_query_params(highlight="true")
            st.rerun()

# --- Undo/Redo ---
col3, col4 = st.columns(2)
with col3:
    if st.button("↩️ Undo"):
        if len(st.session_state.history) > 1:
            last = st.session_state.history.pop()
            st.session_state.future.append(copy.deepcopy(st.session_state.tagged_data))
            st.session_state.tagged_data = copy.deepcopy(st.session_state.history[-1])
            st.session_state.recently_modified = set()
            st.session_state.recently_added_tags = {}
            st.success("🔙 Undid last change.")
        else:
            st.warning("No more undo steps available.")
with col4:
    if st.button("↪️ Redo"):
        if st.session_state.future:
            next_state = st.session_state.future.pop()
            st.session_state.history.append(copy.deepcopy(next_state))
            st.session_state.tagged_data = copy.deepcopy(next_state)
            st.session_state.recently_modified = set()
            st.session_state.recently_added_tags = {}
            st.success("🔁 Redid change.")
        else:
            st.warning("No more redo steps available.")

# --- Undo history display ---
st.markdown("### 🧠 Undo History")
for idx, snapshot in enumerate(st.session_state.history):
    tagged_count = len([v for v in snapshot.values() if v])
    st.markdown(f"- State {idx + 1}: {tagged_count} video(s) tagged")

# --- Highlighted HTML table ---
st.markdown("### 📄 Current Tags")

def render_highlighted_table():
    rows = []
    for title in titles:
        tags = st.session_state.tagged_data[title].split(",")
        tags = [tag.strip() for tag in tags if tag.strip()]
        highlighted = st.session_state.recently_added_tags.get(title, [])

        tag_html = []
        for tag in tags:
            if tag in highlighted:
                tag_html.append(f"<span class='flash-tag'>{tag}</span>")
            else:
                tag_html.append(f"<span>{tag}</span>")

        tag_html_str = ", ".join(tag_html)
        rows.append(f"<tr><td>{title}</td><td>{tag_html_str}</td></tr>")

    table = f"""
    <style>
        .flash-tag {{
            background-color: #ffe599;
            padding: 2px 4px;
            border-radius: 4px;
            animation: fadeOut 4s forwards;
        }}
        @keyframes fadeOut {{
            0% {{ background-color: #ffe599; }}
            100% {{ background-color: transparent; }}
        }}
        table.custom {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        table.custom th, table.custom td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        table.custom th {{
            background-color: #f2f2f2;
        }}
    </style>
    <table class='custom'>
        <thead>
            <tr><th>Title</th><th>Tags</th></tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
    st.markdown(table, unsafe_allow_html=True)

render_highlighted_table()

# --- Clear highlight after 4 seconds (only once) ---
params = st.experimental_get_query_params()
if params.get("highlight") == ["true"]:
    time.sleep(4)
    st.session_state.recently_added_tags = {}
    st.experimental_set_query_params()
    st.rerun()

# --- Export ---
st.markdown("### 💾 Export")
csv_data = pd.DataFrame([
    {"Title": title, "Tags": st.session_state.tagged_data[title]}
    for title in titles
]).to_csv(index=False)

st.download_button("📥 Download CSV", data=csv_data, file_name="tagged_videos.csv", mime="text/csv")
