import os    #to get of directory paths
import glob    #to read all files in directory
import re    #to decompose strings

#### MAIN ####
# user defined parameters
numberOfActivitiesToLoad = 900
numberOfActivities = 30

# read all activity sequences from database
relativePath = os.path.dirname(__file__)
absolutePath = relativePath + "/RG30/Set 1/"
absolutePathRCP = absolutePath + "*.rcp.txt.txt"
files = glob.glob(absolutePathRCP)
print(files)
for i,File in enumerate(files):
    #print (i,File)
    #open file
    with open(File,"r") as f:
        fileName = os.path.basename(f.name)#string
        print(fileName)
        filenumber=re.split(r'[.]',fileName)#from fileName abstract number
        print(filenumber)
        newfilename=filenumber[0]
        # filenumber = list(map(int, filenumber))
        # print(filenumber)
        # if len(filenumber)==3:
        #     fileNumber=filenumber[0]*100+filenumber[1]*10+filenumber[2]
        # elif len(filenumber)==2:
        #     fileNumber = filenumber[0] * 10 + filenumber[1]
        # else:
        #     fileNumber=filenumber[0]
        # print(fileNumber)
        # if fileNumber in range(10,100):
        #     fileNumber = str(fileNumber)
        #     fileNumber = "11" + fileNumber
        # elif fileNumber in range(1,10):
        #     fileNumber = str(fileNumber)
        #     fileNumber = "110" + fileNumber
        # else:
        #     fileNumber=str(fileNumber)
        # print(fileNumber)
        #newFileName = "RG30_"
        #newFileName += fileNumber
        newFileName =newfilename + ".txt"
        print(newFileName)
    os.rename(absolutePath + fileName, absolutePath + newFileName)