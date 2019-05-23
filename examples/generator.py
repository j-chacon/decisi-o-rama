# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:22:05 2019

@author: jchaconhurtado
"""
import sys
sys.path.append('..')

import numpy as np
from decisiorama.utils import random_instance as ri


# action labels
act_labels = ['A1a','A1b','A2','A3','A4','A5','A6','A7','A8a','A8b','A9']
#obj_labels = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 'vol_ffw',
#              'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 'aes_hw', 
#              'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 'no3_dw',
#              'pest', 'bta_dw', 'efqm', 'voice', 'auton', 'time', 'area',
#              'collab', 'cost_cap', 'cost_change',]

_keys = ['rehab', 'adapt', 'gwhh', 'econs', 'vol_dw', 'vol_hw', 'vol_ffw', 
         'reliab_dw', 'reliab_hw', 'reliab_ffw', 'aes_dw', 'aes_hw', 
         'faecal_dw', 'faecal_hw', 'cells_dw', 'cells_hw', 'no3_dw', 'no3_hw',
         'pest_dw', 'pest_hw', 'bta_dw', 'bta_hw', 'efqm', 'voice', 'auton', 'time', 'area', 'collab', 
         'costcap', 'costchange', 'intergen', 'res_gw_prot', 'water_supply', 
         'soc_accept', 'costs', 'dw_supply', 'hw_supply', 'ffw_supply', 
         'dw_quality', 'hw_quality', 'dw_micro_hyg', 'dw_phys_chem', 
         'hw_micro_hyg', 'hw_phys_chem', 'water_supply_IS']

obj_maximise = dict(
        rehab = True,
        adapt = True,
        gwhh = False,
        econs = False,
        vol_dw = False,
        vol_hw = False,
        vol_ffw = True,
        reliab_dw = False,
        reliab_hw = False,
        reliab_ffw = False,
        aes_dw = False,
        aes_hw = False,
        faecal_dw = False,
        faecal_hw = False,
        cells_dw = False,
        cells_hw = False,
        no3_dw = False,
        no3_hw = False,
        pest_dw = False,
        pest_hw = False,
        bta_dw = False,
        bta_hw = False,
        efqm = True,
        voice = True,
        auton = True,
        time = False,
        area = False,
        collab = True,
        costcap = False,
        costchange = False,
        
        # The secondary (higher aggregation) objectives this does not matter
        intergen = False,
        res_gw_prot = False, 
        water_supply = False, 
        soc_accept = False, 
        costs = False, 
        dw_supply = False, 
        hw_supply = False, 
        ffw_supply = False,
        dw_quality = False, 
        hw_quality = False, 
        dw_micro_hyg = False, 
        dw_phys_chem = False, 
        hw_micro_hyg = False, 
        hw_phys_chem = False, 
        water_supply_IS = False,
        )

obj_limits = dict(
        rehab = [0.0, 100.0],
        adapt = [0.0, 100.0],
        gwhh = [0.0, 180.0],
        econs = [0.0, 2.0],
        vol_dw = [0.0, 365.0],
        vol_hw = [0.0, 365.0],
        vol_ffw = [500.0, 3600.0],
        reliab_dw = [0.0, 0.25],
        reliab_hw = [0.0, 0.25],
        reliab_ffw = [0.0, 0.25],
        aes_dw = [0.0, 365.0],
        aes_hw = [0.0, 365.0],
        faecal_dw = [0.0, 365.0],
        faecal_hw = [0.0, 365.0],
        cells_dw = [0.0, 2.0],
        cells_hw = [0.0, 2.0],
        no3_dw = [0.0, 20.0],
        no3_hw = [0.0, 20.0],
        pest_dw = [0.0, 0.02],
        pest_hw = [0.0, 0.02],
        bta_dw = [0.0, 150.0],
        bta_hw = [0.0, 150.0],
        efqm = [20.0, 95.0],
        voice = [0.0, 100.0],
        auton = [0.0, 100.0],
        time = [0.0, 10.0],
        area = [0.0, 10.0],
        collab = [1.0, 6.0],
        costcap = [0.01, 5.0],
        costchange = [0.0, 5.0],
        
        # The secondary (higher aggregation) objectives
        intergen = [-np.inf, np.inf],
        res_gw_prot = [-np.inf, np.inf], 
        water_supply = [-np.inf, np.inf], 
        soc_accept = [-np.inf, np.inf], 
        costs = [-np.inf, np.inf], 
        dw_supply = [-np.inf, np.inf], 
        hw_supply = [-np.inf, np.inf], 
        ffw_supply = [-np.inf, np.inf],
        dw_quality = [-np.inf, np.inf], 
        hw_quality = [-np.inf, np.inf], 
        dw_micro_hyg = [-np.inf, np.inf], 
        dw_phys_chem = [-np.inf, np.inf], 
        hw_micro_hyg = [-np.inf, np.inf], 
        hw_phys_chem = [-np.inf, np.inf], 
        water_supply_IS = [-np.inf, np.inf],
        )


# min, mean, max
_wg_vals = dict(rehab =[0, 0.52, 0.83],	  # rehab
                adapt = [0, 0.38, 0.77],  # adapt
                gwhh = [0.38, 0.73, 1.0],  # gwhh
                econs = [0.0, 0.28,0.63],  # econs
                vol_dw = [0.0, 0.22, 0.36], # vol_dw
                vol_hw = [0.14, 0.28, 0.48], # vol_hw
                vol_ffw = [0.0, 0.34, 0.50], # vol_ffw
                reliab_dw = [0.15, 0.33, 0.48], # reliab_dw
                reliab_hw = [0.26, 0.42, 0.59], # reliab_hw
                reliab_ffw = [0.0, 0.56, 0.83], # reliab_ffw
                aes_dw = [0.07, 0.3, 0.45], # aes_dw
                aes_hw = [0.19, 0.41, 0.83], # aes_hw
                faecal_dw = [0.5, 0.68, 1.0], # faecal_dw  == dw_hygiene
                faecal_hw = [0.5, 0.68, 1.0], # faecal_hw  == hw hygiene
                cells_dw = [0.0, 0.33, 0.50], # cells_dw  = microbial regrowth
                cells_hw = [0.0, 0.32, 0.50], # cells_hw
                no3_dw = [0.0, 0.27, 1.0], # no3_dw  == inorganics
                no3_hw = [0.0, 0.27, 1.0], # no3_dw  == inorganics
                pest_dw = [0.0, 0.34, 1.0], # pest  = pest_dw
                pest_hw = [0.0, 0.34, 1.0], # pest  = pest_dw
                bta_dw = [0.0, 0.42, 1.0], # bta_dw  = micropollutants
                bta_hw = [0.0, 0.42, 1.0], # bta_dw  = micropollutants
                efqm = [0.0, 0.25, 0.83], # efqm  = operational management
                voice = [0.0, 0.11, 0.29], # voice == codetermination
                auton = [0.0, 0.11, 0.33], # auton
                time = [0.0, 0.1, 0.28], # time
                area = [0.0, 0.09, 0.28], # area
                collab = [0.0, 0.14, 0.33], # collab  = unnecesary disturbance
                costcap = [0.23, 0.54, 1.0], # costcap
                costchange = [0.29, 0.33, 0.38], # costchange
                
                # from this point on are the higher level weights
                water_supply_IS = [0.0, 1.0, 2.0],  # dummy value will never be used
                
                # First_level
                intergen = [0.0, 0.19, 0.34],
                res_gw_prot = [0.06, 0.24, 0.48],
                water_supply = [0.23, 0.33, 0.43],
                soc_accept = [0.0, 0.08, 0.23],
                costs = [0.07, 0.18, 0.23],
                
                # second_level
                dw_supply = [0.28, 0.48, 0.83],
                hw_supply = [0.07, 0.29, 0.43],
                ffw_supply = [0.0, 0.24, 0.43],
                
                # Third level
                dw_quality = [0.30, 0.45, 0.83],
                hw_quality = [0.05, 0.31, 0.54],
                
                # fourth level
                dw_micro_hyg = [0.33, 0.44, 0.71],
                dw_phys_chem = [0.05, 0.26, 0.36],
                hw_micro_hyg = [0.07, 0.44, 0.67],
                hw_phys_chem = [0.0, 0.15, 0.36],
            )


def weights(n=None):
    # get weights
    out = {}
    for wg in _wg_vals.keys():
        out[wg] = ri.Truncnormal(_wg_vals[wg][1], 
           (_wg_vals[wg][2] - _wg_vals[wg][0])/3.9, 
           0.0, 1.0, n).get
    return out

#weights(2)

def sq_rehab():
    return status_quo()[0]

def sq_adapt():
    return status_quo()[1]

def status_quo():
    status_quo = dict( rehab = [
        ri.Beta(9.0375, 4.0951).get,
        ri.Beta(9.0375, 4.0951).get,
        ri.Beta(19.0754,8.9788).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Beta(19.0754, 8.9788).get,
        ri.Uniform(0,0).get,
        ri.Normal(0.0438, 0.0162).get,
        ri.Normal(0.0438, 0.0162).get,
        ri.Uniform(0,0).get,
              ], adapt=[
        ri.Normal(35.0, 7.65).get,
        ri.Normal(40.0, 10.2).get,
        ri.Normal(20.0, 10.2).get,
        ri.Normal(85.0, 7.65).get,
        ri.Normal(62.5, 6.38).get,
        ri.Normal(62.5, 6.38).get,
        ri.Normal(55.0, 7.65).get,
        ri.Normal(65.0, 7.65).get,
        ri.Normal(35.0, 7.65).get,
        ri.Normal(35.0, 7.65).get,
        ri.Normal(30.0, 10.2).get,
          ], gwhh = [
        ri.Normal(6.45, 1.08).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(5.32, 0.89).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(11.0, 1.84).get,
        ri.Normal(8.49, 1.42).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(6.45, 1.08).get,
        ri.Normal(6.45, 1.08).get,
            ], econs =[
        ri.Normal(0.713, 0.1783).get,
        ri.Normal(0.713, 0.1783).get,
        ri.Normal(0.713, 0.1783).get,
        ri.Normal(0.0777, 0.0194).get,
        ri.Normal(0.4, 0.1).get,
        ri.Normal(0.3649, 0.0912).get,
        ri.Normal(0.55, 0.1375).get,
        ri.Normal(0.185, 0.0462).get,
        ri.Normal(0.67, 0.1675).get,
        ri.Normal(0.67, 0.1675).get,
        ri.Normal(0.67, 0.1675).get,
            ], vol_dw=[
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
            ], vol_hw =[
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
            ], vol_ffw=[
        ri.Normal(1766.968, 442.0).get,
        ri.Normal(1766.968, 442.0).get,
        ri.Normal(1310.211, 328.0).get,
        ri.Normal(1726.288, 432.0).get,
        ri.Normal(1766.968, 442.0).get,
        ri.Normal(1838.676, 460.0).get,
        ri.Normal(1310.211, 328.0).get,
        ri.Normal(1838.676, 460.0).get,
        ri.Normal(1766.968, 442.0).get,
        ri.Normal(1766.968, 442.0).get,
        ri.Normal(1310.211, 328.0).get,
            ], reliab_dw =[
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Uniform(0.98,1.0).get,
        ri.Normal(0.0827, 0.0161).get,
        ri.Normal(0.175, 0.0375).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Normal(0.065, 0.0175).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.0617, 0.3748).get,
            ], reliab_hw =[
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Normal(0.65, 0.0175).get,
        ri.Lognormal(-4.0617, 0.3748).get,
        ri.Normal(0.175, 0.0375).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Normal(0.065, 0.0175).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.0617, 0.3748).get,
            ],reliab_ffw =[
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.2162, 0.2991).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Normal(0.65, 0.0175).get,
        ri.Lognormal(-4.0617, 0.3748).get,
        ri.Normal(0.175, 0.0375).get,
        ri.Lognormal(-5.1793, 0.3056).get,
        ri.Normal(0.065, 0.0175).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.2198, 0.3378).get,
        ri.Lognormal(-4.0617, 0.3748).get,
            ], aes_dw =[
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(1.0, 0.51).get,
        ri.Normal(1.0, 0.51).get,
        ri.Normal(20.0, 5.1).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(27.5, 11.48).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(10.0, 5.1).get,
            ], aes_hw = [
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(55.0, 22.96).get,
        ri.Normal(75.0, 12.76).get,
        ri.Normal(20.0, 5.1).get,
        ri.Normal(10.0, 5.1).get,
        ri.Normal(27.5, 11.48).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(5.0, 2.55).get,
        ri.Normal(10.0, 5.1).get,            
            ], faecal_dw =[
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Normal(1.0, 0.51).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(5.0, 2.55).get,
            ], faecal_hw =[
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Uniform(0,0).get,
        ri.Normal(20.0, 5.1).get,
        ri.Normal(1.0, 0.51).get,
        ri.Normal(5.0, 2.55).get,
        ri.Uniform(0,0).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(2.5, 1.28).get,
        ri.Normal(5, 2.55).get,    
            ], cells_dw =[
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Normal(0.15, 0.08).get,
        ri.Normal(-0.5, 0.26).get,
        ri.Normal(-1.5, 0.26).get,
        ri.Normal(0.14, 0.07).get,
        ri.Normal(0.34, 0.07).get,
        ri.Normal(0.1, 0.05).get,
        ri.Normal(0.1, 0.05).get,
        ri.Normal(0.15, 0.08).get,
            ], cells_hw =[
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Normal(0.1, 0.05).get,
        ri.Normal(0.39, 0.05).get,
        ri.Normal(0.35, 0.18).get,
        ri.Normal(-1.5, 0.26).get,
        ri.Normal(0.24, 0.03).get,
        ri.Normal(0.34, 0.07).get,
        ri.Normal(0.1, 0.05).get,
        ri.Normal(0.1, 0.05).get,
        ri.Normal(0.15, 0.08).get,
            ], no3_dw =[
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
            ], no3_hw =[
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
        ri.Uniform(0.0, 20.0).get,
            ],pest_dw =[
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
            ], pest_hw =[
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
        ri.Uniform(0.0, 0.02).get,
            ], bta_dw =[
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
            ], bta_hw =[
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
        ri.Uniform(0.0, 150.0).get,
            ], efqm =[
        ri.Normal(68.0, 6.63).get,
        ri.Normal(72.0, 6.63).get,
        ri.Normal(69.0, 4.59).get,
        ri.Normal(37.0, 5.61).get,
        ri.Normal(39.0, 7.65).get,
        ri.Normal(33.0, 5.61).get,
        ri.Normal(65.0, 2.55).get,
        ri.Normal(62.0, 5.1).get,
        ri.Normal(63.0, 2.55).get,
        ri.Normal(63.0, 2.55).get,
        ri.Normal(46.0, 8.16).get,
            ], voice =[
        ri.Normal(20.0, 10.2).get,
        ri.Normal(40.0, 10.2).get,
        ri.Normal(50.0, 4.51).get,
        ri.Normal(80.0, 10.2).get,
        ri.Normal(70.0, 15.31).get,
        ri.Normal(80.0, 10.2).get,
        ri.Normal(60.0, 10.2).get,
        ri.Normal(75.0, 12.76).get,
        ri.Normal(70.0, 10.2).get,
        ri.Normal(70.0, 10.2).get,
        ri.Normal(80.0, 10.2).get,
            ], auton =[
        ri.Uniform(55.1981, 55.1981).get,
        ri.Uniform(55.2, 55.2).get,
        ri.Uniform(55.2, 55.2).get,
        ri.Uniform(80.32, 80.32).get,
        ri.Uniform(55.46, 55.46).get,
        ri.Uniform(100.0, 100.0).get,
        ri.Uniform(90.0, 90.0).get,
        ri.Uniform(89.33, 89.33).get,
        ri.Uniform(55.46, 55.46).get,
        ri.Uniform(55.4571, 55.4571).get,
        ri.Uniform(55.46, 55.46).get,
            ], time =[
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0.36, 0.36).get,
        ri.Uniform(1.69, 1.69).get,
        ri.Uniform(5.0, 5.0).get,
        ri.Uniform(8.04, 8.04).get,
        ri.Uniform(0.36, 0.36).get,
        ri.Uniform(1.69, 1.69).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
        ri.Uniform(0,0).get,
            ], area =[
        ri.Uniform(0.0, 0.0).get,
        ri.Uniform(0.0, 0.0).get,
        ri.Uniform(0.0, 0.0).get,
        ri.Uniform(7.35, 7.35).get,
        ri.Uniform(0.25, 0.25).get,
        ri.Uniform(5.63, 5.63).get,
        ri.Uniform(6.78, 6.78).get,
        ri.Uniform(7.09, 7.09).get,
        ri.Uniform(0.0, 0.0).get,
        ri.Uniform(0.0, 0.0).get,
        ri.Uniform(0.0, 0.0).get,
            ], collab=[
        ri.Uniform(6.0, 6.0).get,
        ri.Uniform(6.0, 6.0).get,
        ri.Uniform(6.0, 6.0).get,
        ri.Uniform(1.0, 1.0).get,
        ri.Uniform(1.0, 1.0).get,
        ri.Uniform(2.0, 2.0).get,
        ri.Uniform(6.0, 6.0).get,
        ri.Uniform(6.0, 6.0).get,
        ri.Uniform(2.0, 2.0).get,
        ri.Uniform(2.0, 2.0).get,
        ri.Uniform(1.0, 1.0).get,
            ], costcap=[
        ri.Lognormal(-5.1776, 0.1232).get,
        ri.Lognormal(-5.1776, 0.1232).get,
        ri.Truncnormal(0.0039, 0.0006, 0.002, 0.007).get,
        ri.Lognormal(-4.2529, 0.2835).get,
        ri.Lognormal(-5.6495, 0.1676).get,
        ri.Lognormal(-5.0688, 0.3677).get,
        ri.Truncnormal(0.0039, 0.0006, 0.002, 0.006).get,
        ri.Lognormal(-4.7923, 0.2947).get,
        ri.Lognormal(-5.5707, 0.1603).get,
        ri.Lognormal(-5.5707, 0.1603).get,
        ri.Beta(25.88, 8599.462).get,
            ], costchange=[
        ri.Normal(0.0062, 0.0003).get,
        ri.Normal(0.0062, 0.0003).get,
        ri.Normal(0.0043, 0.0002).get,
        ri.Normal(0.0043, 0.0002).get,
        ri.Normal(0.0038, 0.0002).get,
        ri.Normal(0.0074, 0.0004).get,
        ri.Normal(0.0043, 0.0002).get,
        ri.Normal(0.0094, 0.0005).get,
        ri.Normal(0.0042, 0.0002).get,
        ri.Normal(0.0042, 0.0002).get,
        ri.Normal(0.0032, 0.0001).get,
            ], intergen = None,
            res_gw_prot = None, 
            water_supply = None, 
            soc_accept = None, 
            costs = None, 
            dw_supply = None, 
            hw_supply = None, 
            ffw_supply = None,
            dw_quality = None, 
            hw_quality = None, 
            dw_micro_hyg = None, 
            dw_phys_chem = None, 
            hw_micro_hyg = None, 
            hw_phys_chem = None, 
            water_supply_IS = None,
            )
    return status_quo