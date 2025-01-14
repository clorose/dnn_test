# path: D:\DNN_test\src\data_processor.py
def tool_condition(input):
    for i in range(len(input)):
        if input[i, 4] == "unworn":
            input[i, 4] = 0
        else:
            input[i, 4] = 1
    return input


def item_inspection(input):
    for i in range(len(input)):
        if input[i, 5] == "no":
            input[i, 6] = 2
        elif input[i, 5] == "yes" and input[i, 6] == "no":
            input[i, 6] = 1
        elif input[i, 5] == "yes" and input[i, 6] == "yes":
            input[i, 6] = 0
    return input


def machining_process(input):
    for i in range(len(input)):
        if input[i, 47] == "Prep":
            input[i, 47] = 0
        elif input[i, 47] == "Layer 1 Up":
            input[i, 47] = 1
        elif input[i, 47] == "Layer 1 Down":
            input[i, 47] = 2
        elif input[i, 47] == "Layer 2 Up":
            input[i, 47] = 3
        elif input[i, 47] == "Layer 2 Down":
            input[i, 47] = 4
        elif input[i, 47] == "Layer 3 Up":
            input[i, 47] = 5
        elif input[i, 47] == "Layer 3 Down":
            input[i, 47] = 6
        elif input[i, 47] == "Repositioning":
            input[i, 47] = 7
        elif input[i, 47] == "End" or input[i, 47] == "end":
            input[i, 47] = 8
        elif input[i, 47] == "Starting":
            input[i, 47] = 9
    return input
