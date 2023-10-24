


class AlmCov(TheoryCl,SphereMask):
    def __init__(self,):
        super().__init__(lmax,spins,path=None,theory_lmin=2,clname='3x2pt_kids',mask_path=None,circmaskattr=None,prep_wlm=True,lmin=None,lmax=None,maskname='mask')






    # could contain covariance for all pseudo alm kinds and all information about them. Then calculating covariance for xi or Cl will just 
    # inherit from here. 
    # different classes if either theoryCl (-> pure noise) or mask are not given?


    def cell_cube(self):
        c_all = np.zeros((3,3,self.len_l))
        c_all[0,0] = self.ee.copy()
        c_all[0,2] = self.ne.copy()
        c_all[2,0] = self.ne.copy()
        c_all[2,2] = self.nn.copy()
        return c_all



    def calc_matrix():

