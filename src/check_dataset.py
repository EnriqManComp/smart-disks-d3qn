import os

count = 0
path = "D:/Enrique/SmartDot-Pathfinding/dataset/"
subdirectorios = sorted(os.listdir(path), key=lambda x: int(x) if x.isdigit() else x)

missing_data = []
for subdir in subdirectorios:
    
    dir_number = int(subdir)    
    
    if dir_number == count:      
        count+=1
    else:
        missing_data.append(count)
        count+=1

    
print(len(missing_data))
print("Minimo", min(missing_data))

for a in range(len(missing_data)):
    if missing_data[a] < 344_000:        
        print(missing_data[a])
