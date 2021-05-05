import os
from bs4 import BeautifulSoup, Tag, Doctype

xml_path = 'xml/'
api_path = 'api/'
top_doc = 'group__dnnl__api.xml'

def createTOCTree(soup, groupList):
    filename = soup.attrs['id'] + ".rst"
    filepath = api_path + filename
    reST = ""
    header = ""
    description = ""
    for child in soup.children:
        if child.name is not None:
            if child.name == "title":
                header = child.text
            if child.name == "briefdescription":
                description = child.text
    reST += header + "\n" + len(header)*'*' + "\n\n" + description + "\n\n"
    reST += ".. toctree::\n"
    reST += "   :maxdepth: 1\n\n"
    for key, value in groupList.items():
        reST += "   " + key + "\n"
    with open(filepath, 'w') as f:
        f.write(reST)

def getGroupInfo(soup):
    filename = soup.attrs['id'] + ".rst"
    filepath = api_path + filename
    reST = ""
    header = ""
    compoundName = ""
    for child in soup.children:
        if child.name is not None:
            if child.name == "title":
                header = child.text
            if child.name == "compoundname":
                compoundName = child.text
    reST += header + "\n" + len(header)*'*' + "\n"
    reST += """
Contents
   * `Defines <#breathe-section-title-defines>`__
   * `TypeDefs <#breathe-section-title-typedefs>`__
   * `Enums <#breathe-section-title-enums>`__
   * `Functions <#breathe-section-title-functions>`__

"""
    reST += ".. doxygengroup:: " + compoundName + "\n"
    reST += "   :inner:\n"
    reST += "   :members:\n"
    with open(filepath,'w') as f:
        f.write(reST)

def buildGroupInfo(soup):
    filename = soup.attrs['id'] + ".rst"
    filepath = api_path + filename
    reST = ""
    header = ""
    compoundName = ""
    for child in soup.children:
        if child.name is not None:
            if child.name == "title":
                header = child.text
    reST += header + "\n" + len(header)*'*' + "\n\n"
    #Handle classes
    innerClasses = soup.find_all('innerclass')
    if len(innerClasses) > 0:
        reST += 'Classes\n##########\n\n'
        reST += '.. toctree::\n\n'
        for iClass in innerClasses:
            reST += buildClass(iClass)
    reST += '\n'
    members = soup.find_all('memberdef')
    macros = []
    typedefs = []
    enums = []
    functions = []
    other = []
    for member in members:
        kind = member.attrs['kind']
        if kind == "define":
            macros.append(member)
        elif kind == "typedef":
            typedefs.append(member)
        elif kind == "enum":
            enums.append(member)
        elif kind == "function":
            functions.append(member)
        else:
            other.append(member)
    #Handle macros
    if len(macros) > 0:
        reST += 'Macros\n#######\n\n'
        for macro in macros:
            macroName = macro.find('name')
            reST += '.. doxygendefine:: ' + macroName.text + '\n\n'
    #Handle typdefs
    if len(typedefs) > 0:
        reST += 'TypeDefs\n##########\n\n'
        for typedef in typedefs:
            typedefName = typedef.find('name')
            reST += '.. doxygentypedef:: ' + typedefName.text + '\n\n'
    #Handle enums
    if len(enums) > 0:
        reST += 'Enumerations\n#################\n\n'
        for enum in enums:
            enumName = enum.find('name')
            reST += '.. doxygenenum:: ' + enumName.text + '\n\n'
    #Handle functions
    if len(functions) > 0:
        reST += 'Functions\n##############\n\n'
        for func in functions:
            funcName = func.find('name')
            funcArgs = func.find('argsstring')
            reST += '.. doxygenfunction:: ' + funcName.text + funcArgs.text + '\n\n'
    with open(filepath,'w') as f:
        f.write(reST)

def buildClass(theClass):
    structName = theClass.attrs['refid']
    structPath = api_path + structName + ".rst"
    className = theClass.text
    structReST = className + "\n" + len(className)*'*' + "\n\n"
    structReST += ".. doxygenstruct:: " + className + "\n"
    structReST += "   :members: \n\n"
    tocEntry = '   ' + structName + '\n'
    with open(structPath,'w') as f2:
        f2.write(structReST)
    return tocEntry

def doxyWalker(soup, indent):
    if soup.name is not None:
        indent += " "
        for child in soup.children:
            if child.name is not None:
                if child.name == "compounddef":
                    if 'kind' in child.attrs and child.attrs['kind'] == 'group':
                        kidGroups = {}
                        for grand in child.children:
                            if grand.name == "innergroup":
                                if 'refid' in grand.attrs:
                                    kidGroups[grand.attrs['refid']] = grand.text
                                    newSoupPath = grand.attrs['refid'] + '.xml'
                                    newSoup = getSoup(xml_path + newSoupPath)
                                    doxyWalker(newSoup, indent)
                    if len(kidGroups) > 0:
                        createTOCTree(child, kidGroups)
                    else:
                        #getGroupInfo(child)
                        buildGroupInfo(child)
                else:
                    #print(child.name)
                    doxyWalker(child,indent)

def getSoup(path):
    with open(path) as f:
        doxy = f.read()

    soup = BeautifulSoup(doxy,features="html.parser")
    return soup

if not os.path.isdir(api_path):
    os.mkdir(api_path)

soup = getSoup(xml_path + top_doc)
doxyWalker(soup, "")
