import os

outdir = "processed_data\\"
indir = "raw_data\\"

for filename in os.listdir(indir):

    filestream = open(indir + filename, "r")

    fileContents = filestream.read()

    fileContents = fileContents.replace("repeat chorus", "")
    fileContents = fileContents.replace("Repeat Chorus", "")
    fileContents = fileContents.replace("(Chorus)", "")
    fileContents = fileContents.replace("Chorus\n", "")
    fileContents = fileContents.replace("(Refrain)", "")
    fileContents = fileContents.replace("Refrain\n", "")
    fileContents = fileContents.replace("(Repeat)", "")
    fileContents = fileContents.replace("Repeat\n", "")

    fileContents = fileContents.replace("repeat chorus", "")

    fileContents = fileContents.replace("\n\n\n", "\n")

    for i in range(10):
        fileContents = fileContents.replace("\n\n", "\n")

        if fileContents[-1] == "\n":
            fileContents = fileContents[:-1]

    fileContents = fileContents.replace("â€™", "'")
    fileContents = fileContents.replace("â€˜", "'")

    print(filename)
    outstream = open(outdir + filename, "w")

    outstream.write(fileContents)

    outstream.close()
    filestream.close()


with open("tensor.txt", "w+") as outstream:
    for filename in os.listdir(outdir):
        with open(outdir + filename) as file:
            outstream.write(file.read())
            outstream.write("\n")