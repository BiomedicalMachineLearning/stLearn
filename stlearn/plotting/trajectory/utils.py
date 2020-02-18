def checkType(arr, n=2):  
  
    # If the first two and the last two elements  
    # of the array are in increasing order  
    if (arr[0] <= arr[1] and 
        arr[n - 2] <= arr[n - 1]) : 
        return True  
  
    # If the first two and the last two elements  
    # of the array are in decreasing order  
    elif (arr[0] >= arr[1] and 
          arr[n - 2] >= arr[n - 1]) : 
        return False  