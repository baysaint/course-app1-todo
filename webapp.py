import streamlit as st
import functions_todo
todos = functions_todo.get_todos()

def add_todo():
    todo = st.session_state["new_todo"]+"\n"
    todos.append(todo)
    functions_todo.write_todos(todos)


st.title("My Todo App")
st.subheader("This is a todo app.")
st.write("This app is to increase your productivity.")

for todo in todos:
    st.checkbox(todo)

st.text_input(label="", placeholder="Add new todo...",
              on_change=add_todo,key="new_todo")

st.session_state