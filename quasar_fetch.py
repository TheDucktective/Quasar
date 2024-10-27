import os
from astropy.io import fits


class Fetch:

    def __init__(self,
                 name=None,
                 data_name=None,
                 number=None,
                 z_min=None,
                 z_max=None,
                 BAL=None,
                 SN=None,

                 ):

        """

        name : name of the .fits catalog file, default is 'DR16Q_V4.fits', description : https://data.sdss.org/datamodel/files/BOSS_QSO/DR16Q/DR16Q_v4.html
        data_name : name of the output data file, default is 'qso-data.txt'
        number : number of quasar wanted, default is 10, 'all' fetches all quasars
        z_min : minimum value the redshift z can take, default is 0
        z_max : maximum value the redshift z can take, default is 20
        BAL : minimum Broad Absorption Line probability, default is 0
        SN : minimum Signal to Noise ratio, default is 0

        """

        if name is None:
            self.name='DR16Q_V4.fits'
        else:
            self.name = name
        if data_name is None:
            self.data_name = 'qso-data.txt'
        else:
            self.data_name = data_name
        self.folder_path = os.path.join(os.getcwd(), 'data', 'catalog')
        self.file_path = os.path.join(self.folder_path,self.name)
        self.data_path = os.path.join(os.getcwd(), 'data', self.data_name)
        self.data = fits.getdata(self.file_path,1)
        if number == 'all':
            self.number = len(self.data)
        elif number is None:
            self.number = 10
        else:
            self.number = number-1
        if BAL is None:
            self.BAL = 0
        else :
            self.BAL = BAL
        if z_max is None:
            self.z_max = 20
        else:
            self.z_max = z_max
        if z_min is None:
            self.z_min = 0
        else:
            self.z_min = z_min
        if SN is None:
            self.SN = 0
        else:
            self.SN = SN

    def get_data(self):
        """

        Fetch the required data from the .fits catalog and write it in a separate file

        """
        #file = fits.open(self.file_path)
        #self.data = file[1].data
        if self.number is None:
            self.number = len(self.data)
        open(self.data_path, 'w').close()
        j, i = 0, 0
        while i in range(len(self.data)) and j <= self.number:
            #TO DO: make a logical function condition
            if (self.data[i].field('Z')) >= self.z_min and \
                    (self.data[i].field('Z') <= self.z_max) and\
                    (self.data[i].field('BAL_PROB') >= self.BAL) and\
                    (self.data[i].field('SN_MEDIAN_ALL') >= self.SN):
                f = open(self.data_path, "a")
                line = ''
                for elem in self.data[i][0:6]:
                    line = line + str(elem) + ' '
                j = j + 1
                line = line + str(self.data[i].field('Z')) +' '\
                       + str(self.data[i].field('BAL_PROB')) + ' '\
                       + str(self.data[i].field('SN_MEDIAN_ALL'))
                f.write(line + "\n")
                f.close()
            i = i + 1
        self.file = open(self.data_path, 'r')
        print('Checked ',i,' quasars out of ', len(self.data))
        print('Found ',j,' quasars corresponding to criteria')

    def dir_check(self):
        if not os.path.exists(self.folder_path):
            print('Catalog file must be in folder : ', self.folder_path)
            return
        folder_path = os.path.join(self.folder_path, self.name)
        if not os.path.exists(folder_path):
            print('Catalog file ', self.name, ' not found in folder : ', self.folder_path)
            return
        elif os.path.exists(folder_path):
            print('Catalog file located:', self.name)

    def fetch_data(self):
        """

        Fetch the data from the .txt data file created in the get_data() function

        """
        # f = open(self.data_path,'r')*
        data=[]
        for i in range(self.number+1):
            line = self.file.readline()
            line = list(line.removesuffix('\n').split(' '))
            data.append(line)

        return data

'''
exemple:

p = Fetch(name=None, data_name=None, number='all', z_min=2.5, z_max=3.5, BAL=0.5, SN=30)

p.dir_check()
p.get_data()
'''