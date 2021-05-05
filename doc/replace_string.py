import os
# This is needed to remove the DNNL_API string from the xml as it causes Breathe to crash.

findString = "DNNL_API"
replaceString = ""
targetDir = "xml"

fileExtension = ".xml"
files = []

def getFiles():
    for dirpath, dirnames, filenames in os.walk(targetDir):
        for filename in [f for f in filenames if f.endswith(fileExtension)]:
            filePath = os.path.join(dirpath,filename)
            print("replacing strings in " + filePath)
            outdata = None
            with open(filePath) as f:
                read_data = f.read()
                outdata = read_data.replace(findString,replaceString)
            with open(filePath,"w") as f:
                f.write(outdata)

getFiles()

