print("Hello World!")


def function(input):
    return 

arr_input = [2, 4, 0, 15, 3, 26, 2, 8, 10]

def reverse_array (arr):
    for i in range(len(arr)):
        for k in range(len(arr) - i): 
            if arr[k + i] > arr[i]: 
                arr[i], arr[k+i] = arr[k + i], arr[i]
                
    return arr

arr_output = reverse_array(arr_input)
print(arr_output)
