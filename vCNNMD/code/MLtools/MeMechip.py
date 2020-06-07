import os
from datetime import datetime

def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)


def MeMeChip(InputFile, filename):
    """
    调用meme
    :param InputFile: fasta格式的文件
    :return:
    """
    softwarePath = "/home/lijy/anaconda3/bin/meme-chip"
    outputDir = "../meme-chip/"+filename
    mkdir(outputDir)
    tmp_cmd = softwarePath + " "+ InputFile + " " + "-oc "+ outputDir  + " -meme-p 30"
    os.system(tmp_cmd)
    
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    
    
if __name__ == '__main__':

    import sys
    
    InputFilePath = "/lustre/user/lijy/TFBSEnconde/chipSeqFa/"
    # TestLenlist=['wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
    #    'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
    #    'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
    #    'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
    #    'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
    #    'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
    #    'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']
    
    filename = sys.argv[1]
    filePath = InputFilePath + filename + ".fa"
    MeMeChip(filePath, filename)


