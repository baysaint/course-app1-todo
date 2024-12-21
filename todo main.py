import functions_todo
import time

now = time.strftime("%b %d, %Y %H:%M:%S")

print("It is", now)


while True:
    user_action = input("Type add, show, edit, complete or exit: ")
    user_action = user_action.strip()

    if user_action.startswith("add"):
        todo = user_action[4:]

            #file = open("todos.txt","r")
            #todos = file.readlines()
            #file.close()

        #with open("todos.txt", "r") as file:
        #    todos = file.readlines()

        todos = functions_todo.get_todos()

        todos.append(todo + "\n")
            #file = open("todos.txt","w")
            #file.writelines(todos)
            #file.close()

        functions_todo.write_todos(todos)


    elif user_action.startswith("show"):
        #file = open("todos.txt","r")
        #todos = file.readlines()
        #file.close()

        #with open("todos.txt", "r") as file:
        #    todos = file.readlines()

        todos = functions_todo.get_todos()

        # new_todos = []

        # for item in todos:
        #    new_item = item.strip("\n")
        #    new_todos.append(new_item)

        #new_todos = [item.strip("\n") for item in todos]

        for index, item in enumerate(todos):
            item = item.strip("\n")
            row = f"{index+1}-{item}"
            print(row)

    elif user_action.startswith("edit"):
        try:
            number = int(user_action[5:])
            number = number - 1

            #with open("todos.txt", "r") as file:
            #    todos = file.readlines()

            todos = functions_todo.get_todos()

            new_todo = input("Enter new todo: ")
            todos[number] = new_todo + "\n"

            functions_todo.write_todos(todos)

        except ValueError:
            print("Your command is not valid.")
            continue
    elif user_action.startswith("complete"):
        try:
            number = int(user_action[9:])

            #with open("todos.txt", "r") as file:
            #    todos = file.readlines()

            todos = functions_todo.get_todos()

            index = number-1
            todo_to_remove = todos[index].strip("\n")
            todos.pop(index)

            functions_todo.write_todos(todos)

            message = f"Todo {todo_to_remove} was removed from the list"
            print(message)
        except IndexError:
            print("There is no item with that number.")
            continue


    elif user_action.startswith("exit"):
        break
    else:
        print("Command is not valid.")
print("Good day!")



