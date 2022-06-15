# densityPrifile and submm_luminosity functions were slightly adapted from the code developed in Popping et al., 2019

import numpy as np
from datetime import datetime
import pandas as pd
from despotic import zonedcloud
from despotic.chemistry import NL99_GC
from astropy import units as u
from astropy import constants as constants

DMR = 1.# Dust-to-metal ratio
mu_atom = 2.33 #atomic weight

def densityProfile(mass,size,NZONES = 25, ProfileType = 'Plummer'):
    Radii = np.linspace(0*u.pc,size,NZONES+1)
    radii = (Radii[1:]+ Radii[:-1])/2
    dR = np.abs(Radii[1:] - Radii[:-1])
    
    #Plummer
    if ProfileType == 'Plummer':
        Rp = size/10
        density = (3 * mass / (4. * np.pi * Rp**3) * (1. + radii**2/Rp**2)**(-5./2))
        density = ((density /constants.m_p).cgs)[::-1]
    elif ProfileType == 'Powerlaw':
        #Powerlaws
        alpha = 2
        n0 = mass / (4. * np.pi *size**2 * size**(3. - alpha)/(3. - alpha))
        print('n0: '+str(n0))
        rho = n0 * (size/radii)**alpha
        print('rho: '+str(rho))
        density = ((rho /constants.m_p).cgs)[::-1]
        print('density: '+str(density))
    elif ProfileType == 'Logotropic':
        #Logotropic
        density = (mass / (4./3 * np.pi * size**3) / constants.m_p).cgs
        nHext = 2./3 * density
        density = nHext * (size / radii[::-1])

    return density, dR.to(u.cm)

def submm_luminosity(INPUT, NZONES=25, ProfileType = 'Powerlaw',noClump = False):
    Mcloud,Rcloud,Metallicity,RadField,redshift = INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4]
    Mcloud *= u.Msun
    Rcloud *= u.pc

    #Rcloud =  (Pressure/1.e4)**(-0.25) * (Mcloud.value/290)**0.5 * u.pc #specific way of setting sizes as outlined in Popping et al. 2019
    Pressure = 1.e4*(Rcloud.value/((Mcloud.value/290)**0.5))**(-4) #P/kb

    #================================================================
    #The cloud column densities are defined. This is either a cloud with fixed volume density or some distribution
    if ProfileType != 'Fixed':
        density, dR = densityProfile(Mcloud, Rcloud, NZONES, ProfileType = ProfileType)
        density /=mu_atom #atomic weight correction
        column_density_cgs = np.cumsum(density * dR)

        #set up the zoned cloud (a radially stratified cloud)
        gmc = zonedcloud(colDen = column_density_cgs.value)
        #Hard coded by hand to fix some weird zondecloud issues. Because zonedcloud is not so good with non-fixed cloud profiles
        print(gmc)
        gmc._colDen[1:] = column_density_cgs.value
        gmc.colDen = gmc._colDen[1:] - (gmc._colDen[1:] - gmc._colDen[:-1])/2
        gmc._colDen[1:] = column_density_cgs.value #For some reason gmc._colDen gets changed in the previous line... Have to reset
    else:
        #a fixed average density
        density = (Mcloud / (4./3 * np.pi * Rcloud**3) / constants.m_p).cgs
        density /= 2.33 #attomic weight correction
        column_density_cgs = density * Rcloud.to(u.cm)

        gmc = zonedcloud(colDen = np.linspace(column_density_cgs.value/NZONES,column_density_cgs.value,NZONES))
        gmc._colDen = np.linspace(column_density_cgs.value/NZONES,column_density_cgs.value,NZONES+1)
        gmc.colDen = gmc._colDen[1:] - (gmc._colDen[1:] - gmc._colDen[:-1])/2

    gmc.nH = density.value
    gmc.Td = 10.
    gmc.Tg = 10.
    gmc.rad.TradDust = 10.
    gmc.ionRate = 1.e-17*RadField
    gmc.rad.ionRate = 1.e-17*RadField
    gmc.chi = RadField 
    gmc.rad.chi = RadField

    gmc.sigmaD10   = 2.0e-26  * Metallicity * DMR       # Cross section to 10K thermal radiation, cm^2 H^-1
    gmc.sigmaDPE   = 1.0e-21 * Metallicity * DMR       # Cross section to 8-13.6 eV photons, cm^2 H^-1
    gmc.sigmaDISRF = 3.0e-22  * Metallicity  * DMR      # Cross section to ISRF photons, cm^2 H^-1
    gmc.Zd      = 1.0  * Metallicity * DMR        # Dust abundance relative to solar
    gmc.alphaGD	   = 3.2e-34 * Metallicity * DMR	      # Dust-gas coupling coefficient, erg cm^3 K^-3/2

    gmc.dust.sigmaD10   = 2.0e-26  * Metallicity * DMR       # Cross section to 10K thermal radiation, cm^2 H^-1
    gmc.dust.sigmaDPE   = 1.0e-21 * Metallicity * DMR       # Cross section to 8-13.6 eV photons, cm^2 H^-1
    gmc.dust.sigmaDISRF = 3.0e-22  * Metallicity  * DMR      # Cross section to ISRF photons, cm^2 H^-1
    gmc.dust.Zd      = 1.0  * Metallicity * DMR        # Dust abundance relative to solar
    gmc.dust.alphaGD	   = 3.2e-34 * Metallicity * DMR	      # Dust-gas coupling coefficient, erg cm^3 K^-3/2

    gmc.addEmitter('c+',1.e-100*Metallicity)
    gmc.addEmitter('c',2.e-4*Metallicity)
    gmc.addEmitter('o', 4.e-4*Metallicity)
    gmc.addEmitter('co',1.e-100*Metallicity) 

    gmc.TCMB = 2.73 * (1. + redshift)
    gmc.rad.TCMB = 2.73 * (1. + redshift)

    for nz in range(NZONES):
        gmc.comp[nz].xH2 = 0.5
        gmc.comp[nz].xHe = 0.1
        gmc.emitters[nz]['co'].extrap = True
        gmc.emitters[nz]['c+'].extrap = True
        gmc.emitters[nz]['o'].extrap = True
        gmc.emitters[nz]['c'].extrap = True

    gmc_xH_zones = np.zeros(NZONES)
    gmc_xH2_zones = np.zeros(NZONES)
    gmc_xCO_zones = np.zeros(NZONES)
    gmc_xCp_zones = np.zeros(NZONES)
    gmc_xC_zones = np.zeros(NZONES)
    gmc_xO_zones = np.zeros(NZONES)
    gmc_xOHx_zones = np.zeros(NZONES)
    gmc_radial_column_densities = np.zeros(NZONES)
    cumulative_gmc_masses = np.zeros(NZONES)
    gmc_radius = np.zeros(NZONES)
    H2_rates = []
    
    gmc.setVirial() #set the cloud to virial properties
    gmc.setChemEq(network=NL99_GC, evolveTemp = 'iterateDust', verbose=True, info = {'xC': 2.e-4*Metallicity,'xO':4.e-4*Metallicity}, tol = 1.e-3)
    #gmc.setChemEq(network=NL99_GC, evolveTemp = 'iterateDust', verbose=True, info = {'xC': 2.e-4*Metallicity,'xO':4.e-4*Metallicity})
    
    #save the abundances for each zone
    for nz in range(NZONES):
        gmc_xH_zones[nz] = gmc.chemabundances_zone[nz]['H']
        gmc_xH2_zones[nz] = gmc.chemabundances_zone[nz]['H2']
        gmc_xCO_zones[nz] = gmc.chemabundances_zone[nz]['CO']
        gmc_xCp_zones[nz] = gmc.chemabundances_zone[nz]['C+']
        gmc_xC_zones[nz] = gmc.chemabundances_zone[nz]['C']
        gmc_xO_zones[nz] = gmc.chemabundances_zone[nz]['O']
        gmc_xOHx_zones[nz] = gmc.chemabundances_zone[nz]['OHx']
                        
        gmc_radial_column_densities[nz] = gmc.colDen[nz]
        cumulative_gmc_masses[nz] =  np.sum(gmc.mass()[0:nz+1])/np.sum(gmc.mass())
        gmc_radius[nz] = gmc.radius()[nz]

    #THe H2 fraction of the cloud
    fH2 = np.sum(gmc.mass() * (2. * gmc_xH2_zones))/np.sum(gmc.mass())

    #Number of H2 nuclei
    H2_n_h_nuclei =  np.sum(gmc.mass()*u.g / (constants.m_p * gmc.muH ).to('g')) #Note that mass in zonedcloud is already only the hydrogen atoms

    #CII
    H2_rates.append(gmc.dEdt(fixedLevPop=False,noClump = noClump)) #get the cooling rates: note, I've already calculated the level populations before using LineLum, so take off the fixedlevpop if you haven't done that
    H2_LambdaCp = np.array([r['LambdaLine']['c+'] for r in H2_rates]) #just pull out the line cooling rate for C+
    H2_lcii = (H2_LambdaCp * H2_n_h_nuclei).decompose()*u.erg/u.s #LambdaCp was the cooling rate per h nucleon.   so i multiply by H2_n_h_nuclei which is my number of H per cloud, then by the number of the clouds in the galaxy

    if np.isnan(H2_lcii): H2_lcii = 0.0 *u.erg/u.s

    #CO
    gmclines = gmc.lineLum('co', noClump = noClump)
    CO10 =  (gmclines[0]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO21 =  (gmclines[1]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO32 =  (gmclines[2]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO43 =  (gmclines[3]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO54 =  (gmclines[4]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO65 =  (gmclines[5]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO76 =  (gmclines[6]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO87 =  (gmclines[7]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CO98 =  (gmclines[8]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    if CO10.value <0: CO10*=0
    if CO21.value <0: CO21*=0
    if CO32.value <0: CO32*=0
    if CO43.value <0: CO43*=0
    if CO54.value <0: CO54*=0
    if CO65.value <0: CO65*=0
    if CO76.value <0: CO76*=0
    if CO87.value <0: CO87*=0
    if CO98.value <0: CO98*=0
        
    # KG update: intTBs
        
    CO10_intTB =  gmclines[0]['intTB']
    CO21_intTB =  gmclines[1]['intTB']
    CO32_intTB =  gmclines[2]['intTB']
    CO43_intTB =  gmclines[3]['intTB']
    CO54_intTB =  gmclines[4]['intTB']
    CO65_intTB =  gmclines[5]['intTB']
    CO76_intTB =  gmclines[6]['intTB']
    CO87_intTB =  gmclines[7]['intTB']
    CO98_intTB =  gmclines[8]['intTB']

    gmclines = gmc.lineLum('c', noClump = noClump)
    CI10 =  (gmclines[0]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    CI21 =  (gmclines[1]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s

    gmclines = gmc.lineLum('o', noClump = noClump)
    OI1 =  (gmclines[0]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    OI2 =  (gmclines[1]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
    OI3 =  (gmclines[2]['lumPerH'] * H2_n_h_nuclei).decompose()*u.erg/u.s
        
    return np.array([Mcloud.value, Rcloud.value, Metallicity, RadField, redshift,H2_lcii.value[0],CO10.value, CO21.value, CO32.value, CO43.value, CO54.value, CO10_intTB, CO21_intTB, CO32_intTB, CO43_intTB, CO54_intTB, CI10.value, CI21.value, CO65.value, CO76.value, CO87.value, CO98.value, CO65_intTB, CO76_intTB, CO87_intTB, CO98_intTB, OI1.value, OI2.value, OI3.value, fH2])

def creating_table(gal_id,cloud_list,df_basic,date):

    t1=datetime.now()
    
    df = pd.DataFrame({'Galaxy_ID':[], 'Cloud_ID':[], 'Mcloud':[], 'Rcloud':[], 'Metallicity':[], 'RadField':[], 'redshift':[], 'H2_lcii':[], 'CO10':[], 'CO21':[], 'CO32':[], 'CO43':[], 'CO54':[], 'CO10_intTB':[], 'CO21_intTB':[], 'CO32_intTB':[], 'CO43_intTB':[], 'CO54_intTB':[], 'CI10':[], 'CI21':[], 'CO65':[], 'CO76':[], 'CO87':[], 'CO98':[], 'CO65_intTB':[], 'CO76_intTB':[], 'CO87_intTB':[], 'CO98_intTB':[], 'OI1':[], 'OI2':[], 'OI3':[], 'fH2':[]})
    
    #if not os.path.isfile('/orange/narayanan/karolina.garcia/SLURM_'+date+'/lim_df_'+date+'.csv'):
    #    df.to_csv('/orange/narayanan/karolina.garcia/SLURM_'+date+'/lim_df_'+date+'.csv', index = False)

    #fails_metal=0
    #fails_sfr=0

    for c in cloud_list:
            
        #Mcloud = df_basic['c_Mass'][df_basic['c_Index']==c].astype('float')
        #Rcloud = df_basic['c_Radius'][df_basic['c_Index']==c].astype('float')
        #Metallicity = df_basic['c_Metallicity'][df_basic['c_Index']==c].astype('float')
        #RadField = df_basic['g_SFR'][df_basic['c_Index']==c].astype('float')
        #redshift = df_basic['g_Redshift'][df_basic['c_Index']==c].astype('float')

        Mcloud = float(df_basic['c_Mass'][df_basic['c_Index']==c])
        Rcloud = float(df_basic['c_Radius'][df_basic['c_Index']==c])
        Metallicity = float(df_basic['c_Metallicity'][df_basic['c_Index']==c])
        RadField = float(df_basic['g_SFR'][df_basic['c_Index']==c])
        redshift = float(df_basic['g_Redshift'][df_basic['c_Index']==c])
        
        g = df_basic['g_Index'][df_basic['c_Index']==c]

        print('>Radius: '+str(Rcloud)+'\n>Mcloud: '+str(Mcloud)+'\n>Metallicity: '+str(Metallicity)+'\n>RadField: '+str(RadField))

        #if Metallicity > 1e-3 and Pressure > 100: #see if I can run without these requirements
        try:
            out = submm_luminosity([Mcloud,Rcloud,Metallicity,RadField,redshift],NZONES=8)
            df = df.append({'Galaxy_ID':int(gal_id), 'Cloud_ID':int(c), 'Mcloud':out[0], 'Rcloud':out[1], 'Metallicity':out[2], 'RadField':out[3], 'redshift':out[4], 'H2_lcii':out[5], 'CO10':out[6], 'CO21':out[7], 'CO32':out[8], 'CO43':out[9], 'CO54':out[10], 'CO10_intTB':out[11], 'CO21_intTB':out[12], 'CO32_intTB':out[13], 'CO43_intTB':out[14], 'CO54_intTB':out[15], 'CI10':out[16], 'CI21':out[17], 'CO65':out[18], 'CO76':out[19], 'CO87':out[20], 'CO98':out[21], 'CO65_intTB':out[22], 'CO76_intTB':out[23], 'CO87_intTB':out[24], 'CO98_intTB':out[25], 'OI1':out[26], 'OI2':out[27], 'OI3':out[28], 'fH2':out[29]},ignore_index=True)
        except:
            print('despoticError, moving on \n despoticError, moving on \n despoticError, moving on \n despoticError, moving on \n despoticError, moving on \n despoticError, moving on \n')
            print('>>>>>>>> galaxy '+str(g)+', cloud '+str(c)+' FAILED due to despotic')
            df = df.append({'Galaxy_ID':int(gal_id), 'Cloud_ID':int(c), 'Mcloud':-99, 'Rcloud':-99, 'Metallicity':-99, 'RadField':-99, 'redshift':-99, 'H2_lcii':-99, 'CO10':-99, 'CO21':-99, 'CO32':-99, 'CO43':-99, 'CO54':-99, 'CO10_intTB':-99, 'CO21_intTB':-99, 'CO32_intTB':-99, 'CO43_intTB':-99, 'CO54_intTB':-99, 'CI10':-99, 'CI21':-99, 'CO65':-99, 'CO76':-99, 'CO87':-99, 'CO98':-99, 'CO65_intTB':-99, 'CO76_intTB':-99, 'CO87_intTB':-99, 'CO98_intTB':-99, 'OI1':-99, 'OI2':-99, 'OI3':-99, 'fH2':-99},ignore_index=True)
            pass

        new_dtypes = {'Galaxy_ID': int, 'Cloud_ID': int}
        df = df.astype(new_dtypes)

    df.to_csv('../Output_Tables/SLURM_'+date+'/lim_df_'+date+'.csv', index = False, mode='a',header=False)
    print('>>>>>>>> cloud '+str(c)+' DONE')
    #else:
        #fails_metal+=1
        #print('>>>>>>>> galaxy '+str(g)+', cloud '+str(c)+' failed due to metallicity')

    t2=datetime.now()

    #print('Execution time for galaxy '+str(galaxy_range)+' = '+str(t2-t1))
    #print('Number of sfr fails: '+str(fails_sfr))
    #print('Number of metallicity fails: '+str(fails_metal))
