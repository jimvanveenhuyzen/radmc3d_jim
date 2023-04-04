## A class to setup RADMC-3D models - Original version belongs to Ardjan Sturm.

## ===== Imports:
import glob, os, shutil, sys
import numpy as np
import pandas as pd
import itertools
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as Collections
import matplotlib.patches as Patches
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FixedLocator, AutoMinorLocator

from astropy import units as u
from astropy import constants as c
from astropy.io import fits
from astropy.visualization import simple_norm as SN
from scipy.interpolate import griddata
from scipy.interpolate import interp1d as I1d

from dataclasses import make_dataclass

from radmc3dPy import *
from radmc3d_tools import *

AU      = natconst.au
AUCM    = c.au.to(u.cm).value
cMicron = c.c.to(u.micron/u.s).value

## =====


class Model():
    def __init__(self,
                 Params_file = None,
                 Output_dir  = None,
                 Index       = 0):
        
        ## Setting the class inputs.
        if Params_file is not None:
            self.Params_file = Params_file
            pass
        if Output_dir is not None:
            self.Output_dir = Output_dir
            pass
        self.Index = Index
        
       	if (Params_file is not None) * (Output_dir is not None):
       	    self.Params_init()
       	    pass
       	    
    def Params_init(self,
                    Params_file = None,
                    Output_dir  = None):
       
        ## Overwrites the initial values, if filename and output-folder are given:
        if Params_file is not None:
            self.Params_file = Params_file
            pass
        if Output_dir is not None:
            self.Output_dir = Output_dir
            pass
            
        ## Checks if the outputfolder exists.
        ## If not, it creates the folder.
        if not os.path.exists(self.Output_dir):
            os.system(f'mkdir {self.Output_dir}')
            pass
        
        ## Setups the RADMC-3D parameters:
        ## It checks whether it is a string (e.g., .csv file) or a Pandas Series/DataFrame.
        if isinstance(self.Params_file, str):
            self.Params_radmc3d = pd.read_csv(self.Params_file,
                                              index_col=0)
            #self.Params_radmc3d.drop(columns='DALI_comments',
            #                         inplace=True,
            #                         errors='ignore')
            pass
        elif isinstance(self.Params_file, pd.Series):
            self.Params_radmc3d = self.Params_file.to_frame()
            pass
        elif isinstance(self.Params_file, pd.DataFrame):
            self.Params_radmc3d = self.Params_file
            pass
        else:
            raise AttributeError("Params_radmc3d is not of the correct dtype")
        
        ## Make the parameters numeric, as much as is possible.
        self.Params_radmc3d = self.Params_radmc3d.applymap(pd.to_numeric, 
                                                           errors='ignore')
        self.Params_radmc3d = self.Params_radmc3d.fillna("")
        for Key, Value in zip(self.Params_radmc3d.index,
                              self.Params_radmc3d.values):
            try:
                float(Value[self.Index])
                if Value[0] == "":
                    setattr(self, Key, Value[self.Index])
                    pass
                else:
                    setattr(self, Key, Value[self.Index]*u.Unit(Value[0]))
                    pass
            except:
                if Key == 'Opac_files':
                    setattr(self, Key, Value[self.Index])
                    pass
                elif Value[1] == "calc":
                    setattr(self, Key, Value[self.Index])
                    pass
                else:
                    if Value[0] == "":
                        setattr(self, Key, Value[self.Index])
                        pass
                    else:
                        setattr(self, Key, Value[self.Index]*u.Unit(Value[0]))
                        pass
            pass
        
        if self.Rs == 'calc':
            self.Rs = np.sqrt((self.Ls)/(4*np.pi*c.sigma_sb*self.Ts**(4.))).to(u.R_sun)
            pass
        
        ## Fraction large (FL) and settling (Chi):
        ## If FL = 0, there will be no settling; as then it is only assumed that the model consists of small dust particles.
        ## IF FL > 0, the settling parameter will have the input value.
        if self.FL == 0:
            self.Chi = 0
            pass
        else:
            pass
        return self
        
    def Make_Wavelengths(self, n12, n23, n34, Lam1=0.01, Lam2=2, Lam3=13, Lam4=1e4):
        """
        This function makes a grid of wavelengths, separated on a log scale.
        There are currently three intervals: [0.01, 2], [2, 13] and [13, 1e4].
        """
        
        Lam12       = np.logspace(np.log10(Lam1),
                                  np.log10(Lam2),
                                  n12, endpoint=False)
        Lam23       = np.logspace(np.log10(Lam2),
                                  np.log10(Lam3),
                                  n23, endpoint=False)
        Lam34       = np.logspace(np.log10(Lam3),
                                  np.log10(Lam4),
                                  n34, endpoint=False)
        Wavelengths = np.concatenate([Lam12, Lam23, Lam34])*u.micron
        return Wavelengths
   
    @u.quantity_input ## Checks of the input has correct Astropy units.
    def BlackBody_Freq(self, Freq:u.Hz, Temp:u.K):
        """
        Planck law [Frequency] given an input frequency and temperature.
        """
        Exp = np.exp((c.h*Freq)/(c.k_B*Temp))
        return (2.*c.h*Freq**(3.))/((c.c**(2.)*(Exp-1.)))/u.sr
        
    def BlackBody_Total(self):
        """
        Calculates the total SED from the star. This includes UV accretion.
        """
        self.BB_norm = self.BlackBody_Freq(self.Freq, self.Ts)*np.pi*self.Ls/(c.sigma_sb*self.Ts**(4.))
        
        ## Add UV accretion:
        if self.Lacc > 0:
            self.BB_norm += (self.BlackBody_Freq(self.Freq, self.T_acc)*np.pi*self.Lacc/(c.sigma_sb*self.Tacc**(4.)))
            pass
        elif self.Macc > 0:
            A_acc         = (c.G*self.Ms/self.Rs)*self.Macc*(1./(c.sigma_sb*T_acc**(4.)))
            self.BB_norm += A_acc*np.pi*self.BlackBody_Freq(self.Freq, self.Tacc)
            pass
        
        if self.Lwall > 0:
            self.BB_norm += (self.BlackBody_Freq(self.Freq, self.Twall)*np.pi*self.Lwall/(c.sigma_sb*self.Twall**(4.)))
            pass

        ## Rewriting the input, such that it is scaled to a distance of 1 pc:
        self.StellarSpec = self.BB_norm/((4.*np.pi/u.sr)*(1.*u.pc)**(2.))
        return self
        
    def Make_Disk(self):
        def Grid_Refine_IE_X(X_init, nlev, nspan):
            """
            Refinement of the inner edge of the grid (R-coordinates).
            """
            X   = X_init.copy()
            Rev = X[0]>X[1]
            for ilev in range(nlev):
                X_new = 0.5*(X[1:nspan+1]+X[:nspan])
                X_ref = np.hstack((X, X_new))
                X_ref.sort()
                X     = X_ref
                if Rev:
                    X = X[::-1]
                    pass
                pass
            return X
            
        def Grid_Refine_IE_Z(Z_init, nlev, nspan):
            """
            Refinement of the inner edge of the grid (midplane, Theta-coordinates)s.
            """
            Z     = Z_init.copy()
            Rev   = Z[0]>Z[1]
            for ilev in range(nlev):
                iHalf = int(0.5*(len(Z)-1))
                Z_new1 = 0.5*(Z[iHalf+1:iHalf+nspan+1]+Z[iHalf:iHalf+nspan])
                Z_new2 = 0.5*(Z[iHalf-nspan:iHalf]+Z[iHalf-nspan+1:iHalf+1])
                Z_ref = np.hstack((Z, Z_new1, Z_new2))
                Z_ref.sort()
                Z     = Z_ref
                if Rev:
                    Z = Z[::-1]
                    pass
                pass
            return Z
        
        if self.Rin == 'calc':
            self.Rin = 0.07*np.sqrt(self.Ls.to(u.Lsun).value)*u.au
            pass
        
        ## Radial-grid:
        RR = np.linspace(self.Rin.value, self.Rout.value, self.nr+1)
        if self.UseGRX:
            RR  = Grid_Refine_IE_X(RR, self.GRX_cycles, self.GRX_cells)*u.au
            pass
        else:
            RR *= u.au
            pass
        RR_mid  = 0.5*(RR[:-1]+RR[1:])
        self.nr = len(RR_mid) ## Resets the number of X-gridpoints, following the refinement.
        
        ## === Surface Density
        SDProfile       = (RR_mid/self.Rc)**(-self.Gamma)*np.exp(-(RR_mid/self.Rc)**(2-self.Gamma))
        
        ## Add cavity in surface density:
        Mask             = (RR_mid >= self.RCi) & (RR_mid <= self.RCo)
        SDProfile[Mask] *= self.CDF
        
        ## Find a value 'Sgas' for which the mass will be close to dust mass.
        Sgas = np.arange(1e-10, 1e-5, 1e-10)
        
        MR   = np.asarray([SDProfile[i]*np.pi*((RR[i+1].value)**(2.)-(RR[i].value)**(2.)) for i in range(len(RR_mid))])
        Mass = np.asarray([np.nansum(SG*MR) for SG in Sgas])*u.Msun
        IDX  = np.argmin(abs(self.Mgas-Mass))
        print(f'Mass check: {Mass[IDX]} {self.Mgas} ({Sgas[IDX]})')
               
        SDP = (SDProfile*Sgas[IDX])/self.gdr * (u.Msun/u.au**(2.))
        
        ## === Other setup
        ## Pressure scale height:
        HR           = self.Hc*(RR_mid/self.Rc)**(self.Psi)
        add_Hrim     = self.Hrim*np.exp(-(RR_mid-self.Rin)/(self.Rin))
        HR          += add_Hrim
        
        ## Theta-grid:
        Thetaup  = np.pi/2.-np.arctan(HR.max()*self.MaxZR).value
        TT       = np.linspace(Thetaup, np.pi-Thetaup, 2*int(self.nt)+1)
        if self.UseGRZ:
            TT   = Grid_Refine_IE_Z(TT, self.GRZ_cycles, self.GRZ_cells)
            pass
        TT[0]    = 0.
        TT[-1]   = np.pi
        TT_mid   = 0.5*(TT[:-1]+TT[1:])
        self.nt  = len(TT_mid)
                
        ## Phi-grid:
        PP = np.linspace(0., 2*np.pi, 2)
        
        ## Create the full grid:
        GRR, GTT = np.meshgrid(RR_mid.to(u.au).value, TT_mid)
        GRR     *= u.au
        
        ## Setting up the density for both small and large dust grain species separately:
        ## Including two seperate settlings between inner and outer components.
        RhoD_small       = ((1-self.FL)*SDP/(np.sqrt(2.*np.pi)*GRR*HR)*np.exp(-0.5*(np.tan(0.5*np.pi-GTT)/HR)**(2.)))
        RhoD_large       = self.FL*SDP/(np.sqrt(2.*np.pi)*GRR*HR*self.Chi)*np.exp(-0.5*(np.tan(0.5*np.pi-GTT)/(HR*self.Chi))**(2.))
        
        self.FL_percell                     = RhoD_large/(RhoD_small+RhoD_large)
        self.FL_percell[self.FL_percell>1.] = 1.
        
        #RhoD_gas = RhoD_small*self.gdr/(1.-self.FL)
        RhoD_gas = RhoD_large*self.gdr
        
        ## Set all the parameters:
        self.RhoD_small    = RhoD_small
        self.RhoD_large    = RhoD_large
        self.RhoD_gas      = RhoD_gas
        self.RhoD_tot      = RhoD_small+RhoD_large
        self.gdr_percell   = self.RhoD_gas/self.RhoD_tot
        self.RR            = RR
        self.RR_mid        = RR_mid
        self.TT            = TT
        self.TT_mid        = TT_mid
        self.GRR, self.GZR = np.meshgrid((self.RR_mid.to(u.AU)).value,
                                         np.tan((np.pi/2.-self.TT_mid)*u.rad))
        self.PP            = PP
        return self
        
    def PlotModel(self):
        def Dereddening(Data):
            ## Wavelength must be given in micron!
            ## Flux must be in either erg s-1 cm-2 Hz-1 or erg s-1 cm-2 micron-1!
            ## AV must be given in magnitudes!
            Wavelength, Flux = Data[0], Data[1]
            if len(Data) > 2:
                eFlux = Data[2]
                pass

            if self.Av.value <= 3:
                AtoAv  = 3.55
                WL, EL = np.loadtxt('./Data/Mathis_CCM89_ext.txt', skiprows=8).T
                pass
            elif self.Av.value < 8:
                AtoAv  = 7.75
                WL, EL = np.loadtxt('./Data/extinc3_8.txt', usecols=[0,1]).T
                pass
            else:
                AtoAv  = 7.75
                WL, EL = np.loadtxt('./Data/extinc3_8.txt', usecols=[0,2]).T
                pass

            Freq  = c.c.to(u.micron/u.s)/Wavelength
            IntEL = I1d(WL, EL, fill_value='extrapolate')(Wavelength) 
            AWav  = IntEL*(self.Av.value/AtoAv)   ## Extinction at each wavelength
            Fac   = 10**(0.4*AWav)     ## Correction factor

            Mask          = (Wavelength <= 40)
            CFlux         = Flux.copy()  ## Corrected flux
            CFlux[Mask]  *= Fac[Mask]
            if len(Data) <= 2:
                return CFlux
            else:
                CeFlux        = CFlux * np.sqrt((eFlux/Flux)**(2.))
                CeFlux[~Mask] = eFlux[~Mask]
                return CFlux, CeFlux
    
        ## === SED Data
        ## Observed SED:
        SED = pd.read_csv(self.SEDFile,
                          delimiter=' ', skiprows=3, usecols=[0,1,2,3], header=None,
                          names=['Wavelength', 'Flux', 'eFlux', 'SourceTable'])
        if self.DRSED:
            OFlux, eOFlux = Dereddening([SED['Wavelength'], SED['Flux'], SED['eFlux']])
            pass
        else:
            OFlux, eOFlux = SED['Flux'], SED['eFlux']
            pass
        OFlux  *= SED['Wavelength']
        eOFlux *= SED['Wavelength']
        
        ## Generated SED:
        Spec  = analyze.readSpectrum(fname=f'{self.Output_dir}/spectrum.out').T
        ## Wavelength is given in micron.
        ## Flux is given in erg s-1 cm-2 Hz-1
        MFlux = Spec[1]*(cMicron/Spec[0])*(1/(self.Dist.value**(2.)))  ## Flux to erg s-1 cm-2 and scaled to the distance, from 1 pc to source distance.
        
        ## === Plot both RP and SED:
        fig, ax = plt.subplots(figsize=(7,5))
        
        ax.errorbar(SED['Wavelength'], OFlux, yerr=eOFlux, color='darkslateblue', 
                       ecolor='darkslateblue', fmt='.', capsize=3, label='Observations')
        ax.scatter(Spec[0], MFlux, color='firebrick', s=10, zorder=10, label='Model')
        ax.set(xlabel='Wavelength [micron]', ylabel=r'Flux [erg s$^{-1}$ cm$^{-2}$]',
                  xscale='log', yscale='log', xlim=[1e-1, 2e3], ylim=[5e-15, 1e-8])
                  
        StarSED  = self.StellarSpec.to(u.erg/u.cm**(2.)/u.s/u.Hz)
        StarSED *= (c.c.to(u.micron/u.s)/self.Wavelengths.to(u.micron))
        StarSED *= (1/self.Dist.value**(2.))                 
        ax.plot(self.Wavelengths.to(u.micron), StarSED, color='firebrick', alpha=0.5)
        
        ax.legend(loc='upper right', fontsize=12)
               
        fig.savefig(f'{self.Output_dir}/Figures/Plot-SED.png', dpi=250)
        
        if self.ShowPlots:
            plt.show()
            pass

    ## === A quick function that runs optool, if required:
    def Run_Optool(self, Command):
        os.system(Command)
        pass

    ## === Writing all the necessary files for running RADMC-3D:      
    def Write_DustOpac(self):
        if self.ScatMode > 2: ## Generates the 'dustkapscatmad_{filename}.inp' files.
            ScatExt = '-scat'
            pass
        else: ## This generates the normal 'dustkap_{filename}.inp' files.
            ScatExt = ''
            pass
        ## Use Optool to generate opacities.
        if 'small' in self.Opac_files:
            print('Run optool!')
            Command = f'cd {self.Output_dir} && optool \
                               -c pyr-mg70 0.87 -c c 0.13 \
                               -p {self.Porosity} -dhs 0.8 \
                               -radmc small -amin {self.amin_small.to(u.micron).value} -amax {self.amax_small.to(u.micron).value}\
                               -apow 3.5 -na {self.na} \
                               -lmin 1e-2 -nlam 1000 -chop 3 {ScatExt}'
            self.Run_Optool(Command)
            pass
        if 'large' in self.Opac_files:
            Command = f'cd {self.Output_dir} && optool \
                               -c pyr-mg70 0.87 -c c 0.13 \
                               -p {self.Porosity} -dhs 0.8 \
                               -radmc large -amin {self.amin_large.to(u.micron).value} -amax {self.amax_large.to(u.micron).value} \
                               -apow 3.5 -na {self.na} \
                               -lmin 1e-2 -nlam 1000 -chop 3 {ScatExt}'
            self.Run_Optool(Command)
            pass
        pass
            
        #with open(os.path.join(self.Output_dir, 'dustopac.inp'), 'w+') as f:
        with open(f'{self.Output_dir}/dustopac.inp', 'w+') as f:
            f.write('2\n') #format number, do not change!
            f.write(f'{len(self.Opac_files)}\n')
            f.write('=========================================================\n')
            for OF in self.Opac_files:
                if self.ScatMode > 2:
                    IS = 10 ## Inputstyle
                    pass
                else:
                    IS = 1
                    pass
                f.write(f'{IS}\n')
                f.write('0\n')   #0 is thermal grain, do not need quantum heated grains
                f.write(f'{OF}\n')
                f.write('--------------------------------------------------------\n')
                pass
            pass
        pass
    
    def Write_radmc3dinp(self):        
        #with open(os.path.join(self.Output_dir, 'radmc3d.inp'), 'w+') as f:
        with open(f'{self.Output_dir}/radmc3d.inp', 'w+') as f:
            f.write(f'nphot = {int(self.nphot)}\n')
            f.write(f'scattering_mode_max = {self.ScatMode}\n') 
            f.write(f'iranfreqmode = 1\n')
            f.write(f'setthreads = {int(self.Nthreads)}\n')
            f.write(f'nphot_spec = {int(self.nphot_spec)}\n')
            f.write(f'nphot_scat = {int(self.nphot_spec)}\n')
            f.write(f'dust_2daniso_nphi = {int(self.np)}\n')
            f.write(f'mc_scat_maxtauabs = 30.d0\n')
            f.write(f'modified_random_walk = {self.UseMRW}\n')
            f.write(f"istar_sphere = 1\n")
            pass
        pass
        
    def Write_DustDens(self): ## Write the dust density.
        #with open(os.path.join(self.Output_dir, 'dust_density.inp'), 'w+') as f:
        with open(f'{self.Output_dir}/dust_density.inp', 'w+') as f:
            f.write('1\n')                                     ## Format number
            f.write(f'{np.prod(self.RhoD_small.shape):d}\n')   ## No. of cells
            f.write(f'{len(self.Opac_files)}\n')               ## No. of dust species
            
            if 'small' in self.Opac_files: ## Check if the small grains are included.
                ## Create a 1-D view, fortran-style indexing            
                Data = self.RhoD_small.T.ravel(order='F').to(u.g/u.cm**3).value 
                Data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
                pass
            
            if 'large' in self.Opac_files:
                ## Create a 1-D view, fortran-style indexing
                Data = self.RhoD_large.T.ravel(order='F').to(u.g/u.cm**3).value 
                Data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
                pass
        pass
    
    def Write_Starsinp(self): ## Write the stellar radiation field.
        #with open(os.path.join(self.Output_dir, 'stars.inp'), 'w+') as f: 
        with open(f'{self.Output_dir}/stars.inp', 'w+') as f: 
            f.write('2\n')
            f.write(f'1 {len(self.Wavelengths)}\n')
            f.write(f'{self.Rs.to(u.cm).value:13.6e} {self.Ms.to(u.g).value:13.6e} 0 0 0\n')
            for Value in self.Wavelengths:
                f.write(f'{Value.to(u.micron).value:13.6e}\n') 
                ## Wavelengths must be written in micron.
                pass
            for Value in self.StellarSpec:
                Value = Value.to(u.erg/u.cm**(2.)/u.s/u.Hz).value
                f.write(f'{Value:13.6e}\n')
                pass
            pass
        pass
    
    def Write_Wavelengths(self): ## Write the wavelenght file.
        #with open(os.path.join(self.Output_dir, 'wavelength_micron.inp'), 'w+') as f:
        with open(f'{self.Output_dir}/wavelength_micron.inp', 'w+') as f:
            f.write(f'{len(self.Wavelengths)}\n')
            for Value in self.Wavelengths:
                f.write(f'{Value.to(u.micron).value:13.6e}\n') 
                ## Wavelengths must be written in micron.
                pass
            pass
        pass
        
    def Write_Cam_Wavelengths(self):
        SED = pd.read_csv(self.SEDFile,
                          delimiter=' ', skiprows=3, usecols=[0,1,2,3], header=None,
                          names=['Wavelength', 'Flux', 'eFlux', 'SourceTable']).to_numpy().T[0]
        Arr = np.linspace(SED[0], SED[-1], 100)
        
        if self.MSEDs:
            Wavelengths = np.sort(np.concatenate((SED, Ar)))
            pass
        else:
            Wavelengths = SED
            pass
            
        with open(f'{self.Output_dir}/camera_wavelength_micron.inp','w') as f:
            f.write(f'{len(Wavelengths)}\n')
            for Wavel in Wavelengths:
                f.write(f'{Wavel:13.6e}\n') # needs to be in micron  
                pass
            pass
        pass
        
    def Write_Grid(self): ## Write the grid.
        #with open(os.path.join(self.Output_dir, 'amr_grid.inp'), 'w+') as f:
        with open(f'{self.Output_dir}/amr_grid.inp', 'w+') as f:
            f.write('1\n')    # iformat
            f.write('0\n')    # AMR grid style  (0=regular grid, no AMR)
            f.write('100\n')  # Coordinate system (spherical)
            f.write('0\n')    #
            f.write('1 1 0\n')
            #f.write(f'{self.nr} {self.nt} 1\n')
            f.write(f'{len(self.RR_mid)} {len(self.TT_mid)} 1\n')
            for R in self.RR:
                f.write(f'{(R.to(u.cm)).value:13.10e}\n')
                pass
            for T in self.TT:
                f.write(f'{T:13.10e}\n')
                pass
            for P in self.PP:
                f.write(f'{P:13.6e}\n')
                pass
            f.write('')
            pass
        pass
        
    def Run_RADMC3D(self):        
        os.system(f'cd {self.Output_dir} && radmc3d mctherm')
        os.system(f'cd {self.Output_dir} && radmc3d spectrum incl {self.Inc} posang {self.PA} loadlambda')
        return self
        
    def Create_Setup(self):
        self.Make_Disk()
        self.Wavelengths = self.Make_Wavelengths(1000, 2000, 1000)
        self.Freq        = c.c.to('micron/s')/self.Wavelengths
        self.BlackBody_Total()
        return self
        
    def Write_Setup(self):
        self.Write_Grid()
        self.Write_DustDens()
        self.Write_radmc3dinp()
        self.Write_Starsinp()
        self.Write_DustOpac()
        self.Write_Wavelengths()
        self.Write_Cam_Wavelengths()
        return self
        
    def run_single(self):
        self.Create_Setup()
        self.Write_Setup()
        self.Run_RADMC3D()
        self.PlotModel()
        pass
            
if __name__ == '__main__':  
    Data        = make_dataclass('Data', [('Parameters'), ('Units'), ('Model')])
    Params_file = pd.DataFrame(## === Grid parameters
                               [Data('nr', '', 100), ## Amount of grid points between Rin and Rswap!
                                Data('nt', '', 30),  ## Amount of grid points in the theta (vertical) grid. 
                                Data('np', '', 60),
                                Data('UseGRX', '', 0),      ## Use grid refinement in X-coordinates?		Boolean: 0-No, 1-Yes
                                Data('GRX_cycles', '', 10), ## Grid refinement cycles: Original value = 10
                                Data('GRX_cells', '', 5),   ## No. of cells that will be refined: original value = 5
                                Data('UseGRZ', '', 0),      ## Use grid refinement in Z-coordinates?		Boolean: 0-No, 1-Yes
                                Data('GRZ_cycles', '', 10), ## Grid refinement cycles: Original value = 10
                                Data('GRZ_cells', '', 5),   ## No. of cells that will be refined: original value = 5
                                ## === Disk parameters
                                Data('Rin', 'au', 'calc'),       ## Inner radius of the grid
                                Data('Rout', 'au', 100),    ## Outer radius of the grid
                                Data('Mgas', 'Msun', 2e-2), ## Disk mass; input is gas mass (i.e. with gdr=100 this is 100x the dust mass)
                                Data('gdr', '', 100),       ## Gas-to-dust ratio
                                ## === Cavity setup:
                                Data('RCi', 'au', 0.),  ## Inner radius of cavity; 'calc' will determine dust sublimation radius depending on the stellar luminosity.
                                Data('RCo', 'au', 1.),  ## Outer radius of cavity
                                Data('CDF', '', 1),     ## Cavity depletion factor. 1 = No cavity, 0 = Fully depleted cavity.
                                ## === Component setup.
                                Data('Gamma', '', 1), ## Power-law slope parameter
                                Data('Rc', 'au', 50), ## Characteristic radius
                                ## === Other setup
                                Data('Hc', '', 0.15),  ## Characteristic scale height
                                Data('Psi', '', 0.15), ## Flaring constant
                                Data('Chi', '', 0.1),  ## Settling parameter
                                Data('FL', '', 0.85),  ## Fraction of large grains
                                ## === Additional height parameters
                                Data('Hrim', '', 0),  ## Include an inner rim?
                                Data('MaxZR', '', 6), ## Maximum height-over-radius ratio.
                                ## === Stellar parameters
                                Data('Ms', 'Msun', 1),
                                Data('Rs', 'Rsun', 'calc'), ## 'Calc' uses luminosity-temperature-radius relation to infer stellar radius.
                                Data('Ls', 'Lsun', 1),
                                Data('Ts', 'K', 5800),
                                ## === Accretion parameters; to turn off set Macc, Tacc and Lacc to 0.
                                Data('Macc', 'Msun/yr', 0),  
                                Data('Twall', 'K', 1400),
                                Data('Lwall', 'Lsun', 0),
                                Data('Tacc', 'K', 10000),
                                Data('Lacc', 'Lsun', 0),
                                ## === Run parameters
                                Data('nphot', '', 1e7),
                                Data('Nthreads', '', 10),
                                Data('nphot_spec', '', 1e4),
                                Data('nLam1', '', 100),
                                Data('nLam2', '', 100),
                                Data('nLam3', '', 100), 
                                Data('UseMRW', '', 1),   ## Use the modified random walk method. Boolean: 0-No, 1-Yes
                                ## === SED/Images parameters:
                                Data('SEDFile', '', './Data/'),  ## Spectral Energy Distribution file destination
                                Data('DRSED', '', 1), ## Deredden the SED
                                Data('Av', 'mag', 0), ## Extinction value to use 
                                Data('MSEDs', '', 0), ## Add more points to the SED than only those in the SEDFile (yields a smoother SED, but take longer to run)
                                ## === Show figures?
                                Data('ShowPlots', '', 1),
                                ## ===
                                Data('Dist', 'pc', 100), 
                                Data('Coord', '', '00h00m00s +00d00m00s'), 
                                Data('Inc', '', 1),
                                Data('PA', '', 1+90),     ## NEED to do +90 for the convolution.
                                Data('Lambda', '', 1300), ## Wavelength at which to create continuum radius.
                                Data('NPix', '', 1000),
                                ## === Opacity files & scattering mode.
                                Data('Opac_files', '', ['small', 'large']),
                                Data('ScatMode', '', 10), ## Most realistic scattering option.
                                ## === Optool parameters: --> Set to DIANA opacities.
                                Data('UseOptool', '', 1),  ## Boolean: 0-No, 1-Yes
                                Data('Porosity', '', 0.25),
                                Data('amin_small', 'micron', 0.005),
                                Data('amax_small', 'micron', 1),
                                Data('amin_large', 'micron', 0.005),
                                Data('amax_large', 'micron', 1000), ## 1000 micron = 1 mm = 0.1 cm
                                Data('na', '', 100)])   ## Sample sizes
    Params_file.set_index('Parameters', inplace=True, drop=True)
    Output_dir = f'./Test'
    Remove     = 'Y' 
    ## Remove = 'Y' deletes the output directory
    ## Remove = 'N' raise a warning that the directory already exists
    try:
        os.mkdir(Output_dir)
        os.mkdir(f'{Output_dir}/Figures')
        Params_file.to_csv(f'{Output_dir}/ModelParameters.csv')

        RADMC3D  = Model(Params_file, 
                         Output_dir=Output_dir,
                         Index=1).run_single()
        pass
    except:
        if Remove == 'Y':
            shutil.rmtree(Output_dir)
            os.mkdir(Output_dir)
            os.mkdir(f'{Output_dir}/Figures')
            Params_file.to_csv(f'{Output_dir}/ModelParameters.csv')

            RADMC3D  = Model(Params_file, 
                             Output_dir=Output_dir,
                             Index=1).run_single()
            pass
        elif Remove == 'N':
            warnings.warn('The output directory already exists! Change the folder name or set Remove = "Y"', Warning)
            sys.exit()
            pass
        pass
    pass
pass


























           
