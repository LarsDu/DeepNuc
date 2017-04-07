

class Gff3Parser:
    def __init__(self,filename,feature):
        self.filename = filename
        self.gff_keys = ["seqid","source","type","start","end","score","phase","attributes"]
    
        self.attributes = ["ID", "Name", "Alias", "Parent",
                                "Target", "Gap", "Derives_from",
                                "Note","Dbxref","Ontology_term","Is_circular"]

        self.cur_line=None
        self.fhandle=None
        self.open()
        

    def open(self):
        print "Opening",self.filename
        self.fhandle = open(self.filename,'r')

    def close(self):
        if self.fhandle:
            self.fhandle.close()

    def __iter__(self):
        return self

    def next(self):
        self.cur_line = self.fhandle.readline()
        if self.cur_line == '':
            self.close()
        if not self.cur_line.startswith("#"):
            vals = self.cur_line.strip('\n').split('\t')
            return dict(zip(self.gff_keys,vals))
        else:
            #Recursive call will keep going until an uncommented line is found
            return self.next()    
