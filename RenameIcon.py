import os


severityOffset = {'health':7, 'warning':14, 'critical':21, 'fatal':28}

def renameIcon():
    rootdir = 'C:\\Users\\wyang2\\Desktop\\report'
    dir = os.listdir(rootdir)
    for fileName in dir:
        dirFullPath = os.path.join(rootdir, fileName)
        offset = severityOffset[fileName]

        icons = os.listdir(dirFullPath)
        begin = 0
        for idx, iconName in enumerate(icons):
            nameIndex = int(idx / 2)
            newIndex = offset + nameIndex
            newName = ''
            if 'over' in iconName:
                newName = 'mapPoint' + str(newIndex) + '_over.png'
            elif 'selected' in iconName:
                newName = 'mapPoint' + str(newIndex) + '_selected.png'
            else:
                newName = 'mapPoint' + str(newIndex) + '.png'

            iconOldFullName = os.path.join(dirFullPath, iconName)
            iconNewFullName = os.path.join(rootdir, newName)
            os.rename(iconOldFullName, iconNewFullName)

renameIcon()