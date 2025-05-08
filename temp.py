

def emilyprint2(emfile,emstr):
    with open(emfile, 'a') as f:
        f.write(emstr)

def emilyfile(fname1,fname2):
    fname = fname1
    with open(fname, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 
        firstframe = lines[0:87] 
        lastframe = lines[-86:] 
        teststr = str(firstframe) + "\n" + str(lastframe)
        twostr = teststr.replace('\\n','\n')
        threestr = twostr.replace("', '",'')
        fourstr =  threestr.replace("['",'')
        fivestr =  fourstr.replace("']",'')
    #print (teststr)
        emilyprint2(fname2,fivestr)

sLetters=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
for sChar in sLetters:
    for i in range(1,21):
        sTxtFilename='D:/leap2/temp/' + sChar + '/' + sChar + str(i) + '.txt'
        stxtfile2 = 'D:/leap2/temp/' + sChar + '/' + sChar +  sChar + str(i) + '.txt'
        emilyfile(sTxtFilename,stxtfile2)



