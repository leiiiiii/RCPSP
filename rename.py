import os    #to get of directory paths
import glob    #to read all files in directory
import re    #to decompose strings

#### MAIN ####
# user defined parameters
numberOfActivitiesToLoad = 900
numberOfActivities = 30

# read all activity sequences from database
relativePath = os.path.dirname(__file__)
absolutePath = relativePath + "/RG30/new_set/"
absolutePathRCP = absolutePath + "*.txt"
files = glob.glob(absolutePathRCP)
print(files)
for i,File in enumerate(files):
    #open file
    with open(File,"r") as f:
        fileName = os.path.basename(f.name)#string
        print(fileName)
        filenumber=re.split(r'[.]',fileName)#from fileName abstract number
        print(filenumber)
        newfilename=filenumber[0]
        realnumber = re.findall(r"\d+\.?\d*",newfilename)
        for value in realnumber:
            a = int(value)
            if a <10:
                newfilename = "Pat000" + str(a)
            elif a<100 and a>=10:
                newfilename = "Pat00" + str(a)
            elif a>=100 and a<1000:
                newfilename = "Pat0" + str(a)
            else:
                newfilename = "Pat" + str(a)
            newFileName =newfilename + ".txt"
        print(newFileName)
    os.rename(absolutePath + fileName, absolutePath + newFileName)