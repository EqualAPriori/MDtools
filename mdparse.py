#!/usr/b.in/env python

"""
@package run
Run a MD simulation in OpenMM.  NumPy is required.
Adapted from Lee-Ping Wang's version, hosted on his github at https://github.com/leeping/OpenMM-MD/blob/master/OpenMM-MD.py

Copyright And License
@author Kevin Shen <kevinshen@ucsb.edu>

All code in this file is released under the GNU General Public License.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""


#==================#
#| Global Imports |#
#==================#

import time
from datetime import datetime, timedelta
t0 = time.time()
from ast import literal_eval as leval
import argparse
from xml.etree import ElementTree as ET
import os
import sys
import pickle
import shutil
import numpy as np
from re import sub
from collections import namedtuple, defaultdict, OrderedDict
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
import warnings
# Suppress warnings from PDB reading.
warnings.simplefilter("ignore")
import logging
logging.basicConfig()

#================================#
#       Set up the logger        #
#================================#

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False


#================================#
#       Useful Subroutines       #
#================================#

def GetTime(sec):
    sec = timedelta(seconds=sec)
    d = datetime(1,1,1) + sec
    if d.year > 1:
        return("%dY%02dM%02dd%02dh%02dm%02ds" % (d.year-1, d.month-1, d.day-1, d.hour, d.minute, d.second))
    elif d.month > 1:
        return("%dM%02dd%02dh%02dm%02ds" % (d.month-1, d.day-1, d.hour, d.minute, d.second))
    elif d.day > 1:
        return("%dd%02dh%02dm%02ds" % (d.day-1, d.hour, d.minute, d.second))
    elif d.hour > 0:
        return("%dh%02dm%02ds" % (d.hour, d.minute, d.second))
    elif d.minute > 0:
        return("%dm%02ds" % (d.minute, d.second))
    elif d.second > 0:
        return("%ds" % (d.second))


def EnergyDecomposition(Sim, verbose=False):
    # Before using EnergyDecomposition, make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    Potential = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
    Kinetic = Sim.context.getState(getEnergy=True).getKineticEnergy() / kilojoules_per_mole
    for i in range(Sim.system.getNumForces()):
        EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    EnergyTerms['Potential'] = Potential
    EnergyTerms['Kinetic'] = Kinetic
    EnergyTerms['Total'] = Potential+Kinetic
    return EnergyTerms

def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3):

    """
    Compute the (cross) statistical inefficiency of (two) timeseries.
    Notes
      The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
      The fast method described in Ref [1] is used to compute g.
    References
      [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
      histogram analysis method for the analysis of simulated and parallel tempering simulations.
      JCTC 3(1):26-41, 2007.
    Examples
    Compute statistical inefficiency of timeseries data with known correlation time.
    >>> import timeseries
    >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)
    @param[in] A_n (required, numpy array) - A_n[n] is nth value of
    timeseries A.  Length is deduced from vector.
    @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
    timeseries B.  Length is deduced from vector.  If supplied, the
    cross-correlation of timeseries A and B will be estimated instead of
    the autocorrelation of timeseries A.
    @param[in] fast (optional, boolean) - if True, will use faster (but
    less accurate) method to estimate correlation time, described in
    Ref. [1] (default: False)
    @param[in] mintime (optional, int) - minimum amount of correlation
    function to compute (default: 3) The algorithm terminates after
    computing the correlation time out to mintime when the correlation
    function furst goes negative.  Note that this time may need to be
    increased if there is a strong initial negative peak in the
    correlation function.
    @return g The estimated statistical inefficiency (equal to 1 + 2
    tau, where tau is the correlation time).  We enforce g >= 1.0.
    """

    # Create numpy copies of input arguments.
    A_n = np.array(A_n)
    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)
    # Get the length of the timeseries.
    N = A_n.size
    # Be sure A_n and B_n have the same dimensions.
    if(A_n.shape != B_n.shape):
        raise ParameterError('A_n and B_n must have same dimensions.')
    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0
    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
    # Trap the case where this covariance is zero, and we cannot proceed.
    if(sigma2_AB == 0):
        logger.info('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency')
        return 1.0
    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while (t < N-1):
        # compute normalized fluctuation correlation function at time t
        C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break
        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
        # Increment t and the amount by which we increment t.
        t += increment
        # Increase the interval if "fast mode" is on.
        if fast: increment += 1
    # g must be at least unity
    if (g < 1.0): g = 1.0
    # Return the computed statistical inefficiency.
    return g

def compute_volume(box_vectors):
    """ Compute the total volume of an OpenMM system. """
    [a,b,c] = box_vectors
    A = np.array([a/a.unit, b/a.unit, c/a.unit])
    # Compute volume of parallelepiped.
    volume = np.linalg.det(A) * a.unit**3
    return volume

def compute_mass(system):
    """ Compute the total mass of an OpenMM system. """
    mass = 0.0 * amu
    for i in range(system.getNumParticles()):
        mass += system.getParticleMass(i)
    return mass

def printcool(text,sym="#",bold=False,color=2,ansi=None,bottom='-',minwidth=50):
    """Cool-looking printout for slick formatting of output.
    @param[in] text The string that the printout is based upon.  This function
    will print out the string, ANSI-colored and enclosed in the symbol
    for example:\n
    <tt> ################# </tt>\n
    <tt> ### I am cool ### </tt>\n
    <tt> ################# </tt>
    @param[in] sym The surrounding symbol\n
    @param[in] bold Whether to use bold print
    @param[in] color The ANSI color:\n
    1 red\n
    2 green\n
    3 yellow\n
    4 blue\n
    5 magenta\n
    6 cyan\n
    7 white
    @param[in] bottom The symbol for the bottom bar
    @param[in] minwidth The minimum width for the box, if the text is very short
    then we insert the appropriate number of padding spaces
    @return bar The bottom bar is returned for the user to print later, e.g. to mark off a 'section'
    """
    if logger.getEffectiveLevel() < 20: return
    def newlen(l):
        return len(sub("\x1b\[[0-9;]*m","",l))
    text = text.split('\n')
    width = max(minwidth,max([newlen(line) for line in text]))
    bar = ''.join([sym for i in range(width + 8)])
    print('\n'+bar)
    for line in text:
        padleft = ' ' * int( ((width - newlen(line)) / 2) )
        padright = ' '* (width - newlen(line) - len(padleft))
        if ansi != None:
            ansi = str(ansi)
            print( "%s| \x1b[%sm%s" % (sym, ansi, padleft),line,"%s\x1b[0m |%s" % (padright, sym) )
        elif color != None:
            print( "%s| \x1b[%s9%im%s" % (sym, bold and "1;" or "", color, padleft),line,"%s\x1b[0m |%s" % (padright, sym) )
        else:
            warn_press_key("Inappropriate use of printcool")
    print(bar)
    return sub(sym,bottom,bar)

def printcool_dictionary(Dict,title="General options",bold=False,color=2,keywidth=25,topwidth=50):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.
    The keys in the dictionary are sorted before printing out.
    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    if logger.getEffectiveLevel() < 20: return
    if Dict == None: return
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        # Useful for printing nice-looking dictionaries, i guess.
        #print "\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"'))
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict):
        print( '\n'.join(["%s %s " % (magic_string(str(key)),str(Dict[key])) for key in Dict if Dict[key] != None]) )
    else:
        print( '\n'.join(["%s %s " % (magic_string(str(key)),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] != None]) )
    print( bar )

def bak(fnm):
    """backup existing files to prevent over-writing
    """
    oldfnm = fnm
    if os.path.exists(oldfnm):
        base, ext = os.path.splitext(fnm)
        i = 1
        while os.path.exists(fnm):
            fnm = "%s_%i%s" % (base,i,ext)
            i += 1
        logger.info("Backing up %s -> %s" % (oldfnm, fnm))
        shutil.move(oldfnm,fnm)


#================================#
#  Define custom reporters here  #
#================================#
def EnergyDecomposition(Sim, verbose=False):
    # Before using EnergyDecomposition, make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    Potential = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
    Kinetic = Sim.context.getState(getEnergy=True).getKineticEnergy() / kilojoules_per_mole
    for i in range(Sim.system.getNumForces()):
        EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    EnergyTerms['Potential'] = Potential
    EnergyTerms['Kinetic'] = Kinetic
    EnergyTerms['Total'] = Potential+Kinetic
    return EnergyTerms


class ProgressReport(object):
    def __init__(self, args, file, reportInterval, simulation, total, first=0):
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        self._initial = True
        if self._openedFile:
            self._out = open(file, 'w')
        else:
            self._out = file
        self._interval = args.report_interval * args.timestep * femtosecond
        self._units = OrderedDict()
        self._units['energy'] = kilojoule_per_mole
        self._units['kinetic'] = kilojoule_per_mole
        self._units['potential'] = kilojoule_per_mole
        self._units['temperature'] = kelvin
        if simulation.topology.getUnitCellDimensions() != None :
            self._units['density'] = kilogram / meter**3
            self._units['volume'] = nanometer**3
        self._data = defaultdict(list)
        self._total = total
        # The time step at the creation of this report.
        self._first = first
        self.run_time = 0.0*picosecond
        self.rt00 = 0.0*picosecond
        self.t0 = time.time()
        self.args = args

        ndof = 0
        for ii in range(simulation.system.getNumParticles()): #Added 2019.02.06. From StateDataReporeter, Python
            if simulation.system.getParticleMass(ii) > 0*unit.dalton: #Careful to make sure virtual sites are not included in calculation. loop is slow but only calculated once
                ndof += 3
        ndof -= simulation.system.getNumConstraints()
        if any(type(simulation.system.getForce(ii)) == CMMotionRemover for ii in range(simulation.system.getNumForces())):
            ndof -= 3
        self.ndof = ndof

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, True)

    def analyze(self, simulation):
        PrintDict = OrderedDict()
        for datatype in self._units:
            data   = np.array(self._data[datatype])
            mean   = np.mean(data)
            dmean  = data - mean
            stdev  = np.std(dmean)
            g      = statisticalInefficiency(dmean)
            stderr = np.sqrt(g) * stdev / np.sqrt(len(data))
            acorr  = 0.5*(g-1)*self._interval/picosecond
            # Perform a linear fit.
            x      = np.linspace(0, 1, len(data))
            z      = np.polyfit(x, data, 1)
            p      = np.polyval(z, x)
            # Compute the drift.
            drift  = p[-1] - p[0]
            # Compute the driftless standard deviation.
            stdev1 = np.std(data-p)
            PrintDict[datatype+" (%s)" % self._units[datatype]] = "%13.5f %13.5e %13.5f %13.5f %13.5f %13.5e" % (mean, stdev, stderr, acorr, drift, stdev1)
        printcool_dictionary(PrintDict,"Summary statistics - total simulation time %.3f ps:\n%-26s %13s %13s %13s %13s %13s %13s\n%-26s %13s %13s %13s %13s %13s %13s" % (self.run_time/picosecond,
                                                                                                                                                                          "", "", "", "", "", "", "Stdev",
                                                                                                                                                                          "Quantity", "Mean", "Stdev", "Stderr", "Acorr(ps)", "Drift", "(NoDrift)"),keywidth=30)
    def report(self, simulation, state):
        # Compute total mass in grams.
        mass = compute_mass(simulation.system).in_units_of(gram / mole) /  AVOGADRO_CONSTANT_NA
        # The center-of-mass motion remover subtracts 3 more DoFs
        #ndof = 3*simulation.system.getNumParticles() - simulation.system.getNumConstraints() - 3


        kinetic = state.getKineticEnergy()
        potential = state.getPotentialEnergy() / self._units['potential']
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        temperature = 2.0 * kinetic / kB / self.ndof / self._units['temperature'] #somehow reports wrong... dof prob calculated wrong. use code from stateDataReporter on github instead
        kinetic /= self._units['kinetic']
        energy = kinetic + potential
        pct = 100 * float(simulation.currentStep - self._first) / self._total
        self.run_time = float(simulation.currentStep - self._first) * self.args.timestep * femtosecond
        if pct != 0.0:
            timeleft = (time.time()-self.t0)*(100.0 - pct)/pct
        else:
            timeleft = 0.0

        # Simulation speed is calculated over a single progress report window
        if self.t00 is None:
            nsday = 0.0
        else:
            days = (time.time()-self.t00)/86400
            nsday = (self.run_time - self.rt00) / nanoseconds / days
        self.t00 = time.time()
        self.rt00 = self.run_time

        if simulation.topology.getUnitCellDimensions() != None :
            box_vectors = state.getPeriodicBoxVectors()
            volume = compute_volume(box_vectors) / self._units['volume']
            density = (mass / compute_volume(box_vectors)) / self._units['density']
            if self._initial:
                logger.info("%8s %17s %15s %13s %13s %13s %13s %13s %13s %13s" % ('Progress', 'E.T.A', 'Speed (ns/day)', 'Time(ps)', 'Temp(K)', 'Kin(kJ)', 'Pot(kJ)', 'Ene(kJ)', 'Vol(nm3)', 'Rho(kg/m3)'))
            logger.info("%7.3f%% %17s %15.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f" % (pct, GetTime(timeleft), nsday, self.run_time / picoseconds, temperature, kinetic, potential, energy, volume, density))
            self._data['volume'].append(volume)
            self._data['density'].append(density)
        else:
            if self._initial:
                logger.info("%8s %17s %13s %13s %13s %13s %13s" % ('Progress', 'E.T.A', 'Time(ps)', 'Temp(K)', 'Kin(kJ)', 'Pot(kJ)', 'Ene(kJ)'))
            logger.info("%7.3f%% %17s %13.5f %13.5f %13.5f %13.5f %13.5f" % (pct, GetTime(timeleft), self.run_time / picoseconds, temperature, kinetic, potential, energy))
        self._data['energy'].append(energy)
        self._data['kinetic'].append(kinetic)
        self._data['potential'].append(potential)
        self._data['temperature'].append(temperature)
        self._initial = False

    def __del__(self):
        if self._openedFile:
            self._out.close()



#================================#
#     The input file parser      #
#================================#

class SimulationOptions(object):
    """ Class for parsing the input file. """
    def set_active(self,key,default,typ,doc,allowed=None,depend=True,clash=False,msg=None):
        """ Set one option.  The arguments are:
        key     : The name of the option.
        default : The default value.
        typ     : The type of the value.
        doc     : The documentation string.
        allowed : An optional list of allowed values.
        depend  : A condition that must be True for the option to be activated.
        clash   : A condition that must be False for the option to be activated.
        msg     : A warning that is printed out if the option is not activated.
        """
        doc = sub("\.$","",doc.strip())+"."
        self.Documentation[key] = "%-8s " % ("(" + sub("'>","",sub("<type '","",str(typ)))+")") + doc
        if key in self.UserOptions:
            val = self.UserOptions[key]
        else:
            val = default
        if type(allowed) is list:
            self.Documentation[key] += " Allowed values are %s" % str(allowed)
            if val not in allowed:
                raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not allowed (choose from \x1b[92m%s\x1b[0m)" % (key, str(val), str(allowed)))
        if typ is bool and type(val) == int:
            val = bool(val)
        if val != None and type(val) is not typ:
            raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not the right type (%s required)" % (key, str(val), str(typ)))
        if depend and not clash:
            if key in self.InactiveOptions:
                del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
        else:
            if key in self.ActiveOptions:
                del self.ActiveOptions[key]
            self.InactiveOptions[key] = val
            self.InactiveWarnings[key] = msg

    def force_active(self,key,val=None,msg=None):
        """ Force an option to be active and set it to the provided value,
        regardless of the user input.  There are no safeguards, so use carefully.
        key     : The name of the option.
        val     : The value that the option is being set to.
        msg     : A warning that is printed out if the option is not activated.
        """
        if msg == None:
            msg == "Option forced to active for no given reason."
        if key not in self.ActiveOptions:
            if val == None:
                val = self.InactiveOptions[key]
            del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val != None and self.ActiveOptions[key] != val:
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val == None:
            self.ForcedOptions[key] = self.ActiveOptions[key]
            self.ForcedWarnings[key] = msg + " (Warning: Forced active but it was already active.)"

    def deactivate(self,key,msg=None):
        """ Deactivate one option.  The arguments are:
        key     : The name of the option.
        msg     : A warning that is printed out if the option is not activated.
        """
        if key in self.ActiveOptions:
            self.InactiveOptions[key] = self.ActiveOptions[key]
            del self.ActiveOptions[key]
        self.InactiveWarnings[key] = msg

    def __getattr__(self,key):
        if key in self.ActiveOptions:
            return self.ActiveOptions[key]
        elif key in self.InactiveOptions:
            return None
        else:
            return getattr(super(SimulationOptions,self),key)

    def record(self):
        out = []
        cmd = ' '.join(sys.argv)
        out.append("")
        out.append("Your command was: %s" % cmd)
        out.append("To reproduce / customize your simulation, paste the following text into an input file")
        out.append("and rerun the script with the -I argument (e.g. '-I openmm.in')")
        out.append("")
        out.append("#===========================================#")
        out.append("#|     Input file for OpenMM MD script     |#")
        out.append("#|  Lines beginning with '#' are comments  |#")
        out.append("#===========================================#")
        TopBar = False
        UserSupplied = [] # === determining user-supplied === #
        for key in self.ActiveOptions:
            if key in self.UserOptions and key not in self.ForcedOptions:
                UserSupplied.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(UserSupplied) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|          User-supplied options:         |#")
            out.append("#===========================================#")
            out += UserSupplied
        Forced = [] # === determing forced/overridden by user input === #
        for key in self.ActiveOptions:
            if key in self.ForcedOptions:
                Forced.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
                Forced.append("%-22s %20s # Reason : %s" % ("","",self.ForcedWarnings[key]))
        if len(Forced) > 0:
            if TopBar:
                out.append("#=======================================================================#")
            else:
                TopBar = True
            out.append("#|     Options enforced by the script or overridden by user input:     |#")
            out.append("#=======================================================================#")
            out += Forced
        ActiveDefault = [] # === determining active defaults === #
        for key in self.ActiveOptions:
            if key not in self.UserOptions and key not in self.ForcedOptions:
                ActiveDefault.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(ActiveDefault) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|   Active options at default values:     |#")
            out.append("#===========================================#")
            out += ActiveDefault
        # out.append("")
        out.append("#===========================================#")
        out.append("#|           End of Input File             |#")
        out.append("#===========================================#")
        Deactivated = [] # === determining deactivated === #
        for key in self.InactiveOptions:
            Deactivated.append("%-22s %20s # %s" % (key, str(self.InactiveOptions[key]), self.Documentation[key]))
            Deactivated.append("%-22s %20s # Reason : %s" % ("","",self.InactiveWarnings[key]))
        if len(Deactivated) > 0:
            out.append("")
            out.append("#===========================================#")
            out.append("#|   Deactivated or conflicting options:   |#")
            out.append("#===========================================#")
            out += Deactivated
        Unrecognized = [] # === Unrecognized keys === #
        for key in self.UserOptions:
            if key not in self.ActiveOptions and key not in self.InactiveOptions:
                Unrecognized.append("%-22s %20s" % (key, self.UserOptions[key]))
        if len(Unrecognized) > 0:
            # out.append("")
            out.append("#===========================================#")
            out.append("#|          Unrecognized options:          |#")
            out.append("#===========================================#")
            out += Unrecognized
        return out

    def __init__(self, input_file, overrides={}, pdbfnm=''):
        super(SimulationOptions,self).__init__()
        #basename = os.path.splitext(pdbfnm)[0] #2019.01.31 Removed fancy appending of basename to default outputs
        basename = ''
        self.Documentation = OrderedDict()
        self.UserOptions = OrderedDict()
        self.ActiveOptions = OrderedDict()
        self.ForcedOptions = OrderedDict()
        self.ForcedWarnings = OrderedDict()
        self.InactiveOptions = OrderedDict()
        self.InactiveWarnings = OrderedDict()
        #=== First build a dictionary of user supplied (input file) options. ===#
        if input_file != None:
            for line in open(input_file).readlines():
                line = sub('#.*$','',line.strip())
                s = line.split()
                if len(s) > 0:
                    # Options are case insensitive
                    key = s[0].lower()
                    try:
                        val = leval(line.replace(s[0],'',1).strip())
                    except:
                        val = str(line.replace(s[0],'',1).strip())
                    self.UserOptions[key] = val

	#===  Then include overrides ===#
        for key,val in overrides.items():
            #self.UserOptions[key] = val
            logger.info("Setting key({}) to (%)".format(key,val))            
            if key in self.ActiveOptions: 
                self.UserOptions[key] = val
                #self.force_active(key,val,msg="{} overridden from {} to {} by user".format(key,self.ActiveOptions[key],val))
            else:
                self.UserOptions[key] = val
                #self.force_activate(key,val,msg="{} overridden to {} by user".format(key,val))

        #===            ====            ====            ====            ====      ===#
        #=== Now go through the logic of determining which options are activated. ===#
        #=== Including defaults                                                   ===#
        #===            ====            ====            ====            ====      ===#

        #=== Input and files ===#
        #gromacsdir = '/home/kshen/lib/ff'
        mypath = os.path.abspath(__file__)
        gromacsdir = mypath.split('MDtools')[0]+'ff' 
        self.set_active('topdir',gromacsdir,str,"for gromacs.GROMACS_TOPDIR")
        self.set_active('topfile','system.top',str,"Gromacs system.top file")
        self.set_active('grofile','box.gro',str,"Gromacs .gro file, we just use for box")

        self.set_active('cont',0,int,"continuation flag, for keeping track")
        self.set_active('incoord','in.pdb',str,"input file, must be .pdb or .xml")
        #self.set_active('inpdb','in.pdb',str,"input pdb file, i.e. for getting positions or topology")
        #self.set_active('inxml','in.xml',str,"input xml file, i.e. for continuing simulation")
        #self.set_active('serialize',None,str,"Provide a file name for writing the serialized System object.")

        #=== Reporters ===#
        self.set_active('restart_filename','restart.p',str,"Restart information will be read from / written to this file (will be backed up).")
        self.set_active('read_restart',True,bool,"Restart simulation from the restart file.",
                        depend=(os.path.exists(self.restart_filename)), msg="Cannot restart; file specified by restart_filename does not exist.")
        self.set_active('restart_interval',1000,int,"Specify a timestep interval for writing the restart file.")
        self.set_active('report_interval',100,int,"Number of steps between every progress report.")

        self.set_active('pdb_report_interval',0,int,"Specify a timestep interval for PDB reporter.")
        #self.set_active('pdb_report_filename',"output_%s.pdb" % basename,str,"Specify an file name for writing output PDB file.",
        #                depend=(self.pdb_report_interval > 0), msg="pdb_report_interval needs to be set to a whole number.")

        #self.set_active('dcd_report_interval',0,int,"Specify a timestep interval for DCD reporter.")
        #self.set_active('dcd_report_filename',"output_%s.dcd" % basename,str,"Specify an file name for writing output DCD file.",
        #                depend=(self.dcd_report_interval > 0), msg="dcd_report_interval needs to be set to a whole number.")

        #self.set_active('eda_report_interval',0,int,"Specify a timestep interval for Energy reporter.", clash=(self.integrator=="mtsvvvr"), msg="EDA reporter incompatible with MTS integrator.")
        #self.set_active('eda_report_filename',"output_%s.eda" % basename,str,"Specify an file name for writing output Energy file.",
        #                depend=(self.eda_report_interval > 0), msg="eda_report_interval needs to be set to a whole number.")

        self.set_active('netcdf_report_interval',0,int,"Specify a timestep interval for netcdf reporter.")
        self.set_active('netcdf_vels',False,bool,"Include velocities in netcdf")
        self.set_active('netcdf_frcs',False,bool,"Include forces in netcdf")

        self.set_active('dcd_report_interval',0,int,"Specify a timestep interval for DCD reporter.")

        self.set_active('outpdb','output.pdb',str,"output pdb file", depend=self.pdb_report_interval>0)
        self.set_active('outnetcdf','output.nc',str,"output netcdf file", depend=self.netcdf_report_interval>0)
        self.set_active('outdcd','output.dcd',str,"output dcd file", depend=self.dcd_report_interval>0)
        self.set_active('logfile','thermo.log',str,"log file")

        self.set_active('checkpoint',True,bool,"Flag for turning on checkpoints")
        self.set_active('chkpdb','chk.pdb',str,"checkpoint pdb file",depend=self.checkpoint)
        self.set_active('chkxml','chk.xml',str,"checkpoint xml file",depend=self.checkpoint) 

       
        #=== Runtime ===#
        self.set_active('use_fs_interval',True,bool, "Whether or not to use my convention of setting steps by #femtoseconds.")
        self.set_active('minimize',False,bool,"Specify whether to minimize the energy before running dynamics.")
        self.set_active('timestep',1.0,float,"Time step in femtoseconds.")
        self.set_active('equilibrate',0,int,"Number of steps reserved for equilibration.")
        #self.set_active('production',1000,int,"Number of steps in production run.")
        self.set_active('block_interval',1000000,int,"Number of steps per block")
        self.set_active('nblocks',1,int,"Number of blocks to run")
        self.set_active('checkpoint_interval',1000000,int,"Number of steps between checkpoint.xml files.")
        


        #=== Integrator ===# 
        self.set_active('integrator','verlet',str,"Molecular dynamics integrator",allowed=["verlet","langevin","velocity-verlet","mtsvvvr"])
        self.set_active('temperature',0.0,float,"Simulation temperature for Langevin integrator or Andersen thermostat.")
        if self.temperature <= 0.0 and self.integrator in ["langevin", "mtsvvvr"]:
            raise Exception("You need to set a finite temperature if using the Langevin or MTS-VVVR integrator!")

       
        #=== Handling Pressure ===#
        self.set_active('gentemp',self.temperature,float,"Specify temperature for generating velocities")
        self.set_active('collision_rate',0.1,float,"Collision frequency for Langevin integrator or Andersen thermostat in ps^-1.",
                        depend=(self.integrator in ["langevin", "mtsvvvr"] or self.temperature != 0.0),
                        msg="We're not running a constant temperature simulation")
        self.set_active('pressure',0.0,float,"Simulation pressure; set a positive number to activate.",
                        clash=(self.temperature <= 0.0),
                        msg="For constant pressure simulations, the temperature must be finite")
        self.set_active('anisotropic',False,bool,"Set to True for anisotropic box scaling in NPT simulations",
                        depend=("pressure" in self.ActiveOptions and self.pressure > 0.0), msg = "We're not running a constant pressure simulation")
        self.set_active('nbarostat',25,int,"Step interval for MC barostat volume adjustments.",
                        depend=("pressure" in self.ActiveOptions and self.pressure > 0.0), msg = "We're not running a constant pressure simulation")
        
        #=== Handling Forces ===#
        self.set_active('nonbonded_method','PME',str,"Set the method for nonbonded interactions.", allowed=["NoCutoff","CutoffNonPeriodic","CutoffPeriodic","Ewald","PME","LJPME"])
        self.nonbonded_method_obj = {"NoCutoff":NoCutoff,"CutoffNonPeriodic":CutoffNonPeriodic,"CutoffPeriodic":CutoffPeriodic,"Ewald":Ewald,"PME":PME,"LJPME":LJPME}[self.nonbonded_method]
        self.set_active('nonbonded_cutoff',0.9,float,"Nonbonded cutoff distance in nanometers.")
        self.set_active('vdw_switch',False,bool,"Use a multiplicative switching function to ensure twice-differentiable vdW energies near the cutoff distance.")
        self.set_active('switch_distance',0.8,float,"Set the distance where the switching function starts; must be less than the nonbonded cutoff.",depend=self.vdw_switch)
        self.set_active('dispersion_correction',True,bool,"Isotropic long-range dispersion correction for periodic systems.")
        self.set_active('ewald_error_tolerance',0.0005,float,"Error tolerance for Ewald, PME, LJPME methods.  Don't go below 5e-5 for PME unless running in double precision.",
                        depend=(self.nonbonded_method_obj in [Ewald, PME, LJPME]), msg="Nonbonded method must be set to Ewald or PME or LJPME.")

        #=== Constraints ===#
        self.set_active('initial_report',False,bool,"Perform one Report prior to running any dynamics.")
        self.set_active('constraints',None,str,"Specify constraints.", allowed=[None,"HBonds","AllBonds","HAngles"])
        self.constraint_obj = {None: None, "None":None,"HBonds":HBonds,"HAngles":HAngles,"AllBonds":AllBonds}[self.constraints]
        self.set_active('rigid_water',False,bool,"Add constraints to make water molecules rigid.")
        self.set_active('constraint_tolerance',1e-5,float,"Set the constraint error tolerance in the integrator (default value recommended by Peter Eastman).")

        #=== Platform ===#
        self.set_active('platform',"CUDA",str,"The simulation platform.", allowed=["Reference","CUDA","OpenCL"])
        self.set_active('cuda_precision','single',str,"The precision of the CUDA platform.", allowed=["single","mixed","double"],
                        depend=(self.platform == "CUDA"), msg="The simulation platform needs to be set to CUDA")
        self.set_active('device',None,int,"Specify the device (GPU) number; will default to the fastest available.", depend=(self.platform in ["CUDA", "OpenCL"]), msg="The simulation platform needs to be set to CUDA or OpenCL")

        #=== Check if User input Something that's not used ===#
        for key in self.UserOptions:
            if key not in self.ActiveOptions and key not in self.InactiveOptions:
                logger.info("Unrecognized key: {}".format(key))
                print( "Unrecognized key: {}".format(key) )
 
#End main class

#================================#
#    The command line parser     #
#================================#

# Taken from MSMBulder - it allows for easy addition of arguments and allows "-h" for help.
def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d
    group.add_argument(*args, **kwargs)

'''#=== IF RUNNING A SIMULATION INSTEAD OF USING IT AS A CLASS/Module ===#
print()
print( " #===========================================#" )
print( " #|    OpenMM general purpose simulation    |#" )
print( " #| (Hosted @ github.com/leeping/OpenMM-MD) |#" )
print( " #|  Use the -h argument for detailed help  |#" )
print( " #===========================================#" )
print()

parser = argparse.ArgumentParser()
add_argument(parser, 'pdb', nargs=1, metavar='input.pdb', help='Specify one PDB or AMBER inpcrd file \x1b[1;91m(Required)\x1b[0m', type=str)
add_argument(parser, 'xml', nargs='+', metavar='forcefield.xml', help='Specify multiple force field XML files, one System XML file, or one AMBER prmtop file \x1b[1;91m(Required)\x1b[0m', type=str)
add_argument(parser, '-I', '--inputfile', help='Specify an input file with options in simple two-column format.  This script will autogenerate one for you', default=None, type=str)
cmdline = parser.parse_args()
pdbfnm = cmdline.pdb[0]
xmlfnm = cmdline.xml
args = SimulationOptions(cmdline.inputfile, pdbfnm)


#================================#
#    Create OpenmMM object       #
#================================#
# Create an OpenMM PDB object.
pdb = PDBFile(pdbfnm)

# Detect the presence of periodic boundary conditions in the PDB file.
pbc = pdb.getTopology().getUnitCellDimensions() != None
if pbc:
    logger.info("Detected periodic boundary conditions")
else:
    logger.info("This is a nonperiodic simulation")


#===================================#
#| Recognize System                |#
#| This script uses Gromacs format |#   
#===================================#
from parmed import gromacs
gromacs.GROMACS_TOPDIR = "/home/kshen/SDS"
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
import parmed as pmd

top = gromacs.GromacsTopologyFile(top_file, defines=defines)
gro = gromacs.GromacsGroFile.parse(box_file)
top.box = gro.box


#====================================#
#| Temperature and pressure control |#
#====================================#


def add_barostat():
    if args.pressure <= 0.0:
        logger.info("This is a constant volume (NVT) run")
    elif pbc:
        logger.info("This is a constant pressure (NPT) run at %.2f bar pressure" % args.pressure)
        logger.info("Adding Monte Carlo barostat with volume adjustment interval %i" % args.nbarostat)
        logger.info("Anisotropic box scaling is %s" % ("ON" if args.anisotropic else "OFF"))
        if args.anisotropic:
            logger.info("Only the Z-axis will be adjusted")
            barostat = MonteCarloAnisotropicBarostat(Vec3(args.pressure*bar, args.pressure*bar, args.pressure*bar), args.temperature*kelvin, False, False, True, args.nbarostat)
        else:
            barostat = MonteCarloBarostat(args.pressure * bar, args.temperature * kelvin, args.nbarostat)
        system.addForce(barostat)
    else:
        args.deactivate("pressure", msg="System is nonperiodic")
        #raise Exception('Pressure was specified but the topology contains no periodic box! Exiting...')

def NVEIntegrator():
    if args.integrator == "verlet":
        logger.info("Creating a Leapfrog integrator with %.2f fs timestep." % args.timestep)
        integrator = VerletIntegrator(args.timestep * femtosecond)
    elif args.integrator == "velocity-verlet":
        logger.info("Creating a Velocity Verlet integrator with %.2f fs timestep." % args.timestep)
        integrator = VelocityVerletIntegrator(args.timestep * femtosecond)
    return integrator

def add_thermostat():
    if args.temperature <= 0.0:
        logger.info("This is a constant energy, constant volume (NVE) run.")
        integrator = NVEIntegrator()
    else:
        logger.info("This is a constant temperature run at %.2f K" % args.temperature)
        logger.info("The stochastic thermostat collision frequency is %.2f ps^-1" % args.collision_rate)
        if args.integrator == "langevin":
            logger.info("Creating a Langevin integrator with %.2f fs timestep." % args.timestep)
            integrator = LangevinIntegrator(args.temperature * kelvin, args.collision_rate / picosecond, args.timestep * femtosecond)
        elif args.integrator == "mtsvvvr":
            logger.info("Creating a multiple timestep Langevin integrator with %.2f / %.2f fs outer/inner timestep." % (args.timestep, args.innerstep))
            if int(args.timestep / args.innerstep) != args.timestep / args.innerstep:
                raise Exception("The inner step must be an even subdivision of the time step.")
            integrator = MTSVVVRIntegrator(args.temperature * kelvin, args.collision_rate / picosecond, args.timestep * femtosecond, system, int(args.timestep / args.innerstep))
        else:
            integrator = NVEIntegrator()
            thermostat = AndersenThermostat(args.temperature * kelvin, args.collision_rate / picosecond)
            system.addForce(thermostat)

add_thermostat()            
add_barostat()

if not hasattr(args,'constraints') or (str(args.constraints) == "None" and args.rigidwater == False):
    args.deactivate('constraint_tolerance',"There are no constraints in this system")
else:
    logger.info("Setting constraint tolerance to %.3e" % args.constraint_tolerance)
    integrator.setConstraintTolerance(args.constraint_tolerance)


#==================================#
#|      Create the platform       |#
#==================================#
# if args.platform != None:
logger.info("Setting Platform to %s" % str(args.platform))
try:
    platform = Platform.getPlatformByName(args.platform)
except:
    logger.info("Warning: %s platform not found, going to Reference platform \x1b[91m(slow)\x1b[0m" % args.platform)
    args.force_active('platform',"Reference","The %s platform was not found." % args.platform)
    platform = Platform.getPlatformByName("Reference")

if 'device' in args.ActiveOptions:
    # The device may be set using an environment variable or the input file.
    if os.environ.has_key('CUDA_DEVICE'):
        device = os.environ.get('CUDA_DEVICE',str(args.device))
    elif os.environ.has_key('CUDA_DEVICE_INDEX'):
        device = os.environ.get('CUDA_DEVICE_INDEX',str(args.device))
    else:
        device = str(args.device)
    if device != None:
        logger.info("Setting Device to %s" % str(device))
        #platform.setPropertyDefaultValue("CudaDevice", device)
        platform.setPropertyDefaultValue("CudaDeviceIndex", device)
        #platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
    else:
        logger.info("Using the default (fastest) device")
else:
    logger.info("Using the default (fastest) device")
if "CudaPrecision" in platform.getPropertyNames():
    platform.setPropertyDefaultValue("CudaPrecision", args.cuda_precision)
# else:
#     logger.info("Using the default Platform")


#==================================#
#|  Create the simulation object  |#
#==================================#
logger.info("Creating the Simulation object")
# Get the number of forces and set each force to a different force group number.
nfrc = system.getNumForces()
if args.integrator != 'mtsvvvr':
    for i in range(nfrc):
        system.getForce(i).setForceGroup(i)
for i in range(nfrc):
    # Set vdW switching function manually.
    f = system.getForce(i)
    if f.__class__.__name__ == 'NonbondedForce':
        if 'vdw_switch' in args.ActiveOptions and args.vdw_switch:
            f.setUseSwitchingFunction(True)
            f.setSwitchingDistance(args.switch_distance)
if args.platform != None:
    simulation = Simulation(modeller.topology, system, integrator, platform)
else:
    simulation = Simulation(modeller.topology, system, integrator)
# Serialize the system if we want.
if args.serialize != 'None' and args.serialize != None:
    logger.info("Serializing the system")
    serial = XmlSerializer.serializeSystem(system)
    bak(args.serialize)
    with open(args.serialize,'w') as f: f.write(serial)
# Print out the platform used by the context
printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())

# Print out some more information about the system
logger.info("--== System Information ==--")
logger.info("Number of particles   : %i" % simulation.context.getSystem().getNumParticles())
logger.info("Number of constraints : %i" % simulation.context.getSystem().getNumConstraints())
logger.info("Total system mass     : %.2f amu" % (compute_mass(system)/amu))
for f in simulation.context.getSystem().getForces():
    if f.__class__.__name__ == 'AmoebaMultipoleForce':
        logger.info("AMOEBA PME order      : %i" % f.getPmeBSplineOrder())
        logger.info("AMOEBA PME grid       : %s" % str(f.getPmeGridDimensions()))
    if f.__class__.__name__ == 'NonbondedForce':
        method_names = ["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME"]
        logger.info("Nonbonded method      : %s" % method_names[f.getNonbondedMethod()])
        logger.info("Number of particles   : %i" % f.getNumParticles())
        logger.info("Number of exceptions  : %i" % f.getNumExceptions())
        if f.getNonbondedMethod() > 0:
            logger.info("Nonbonded cutoff      : %.3f nm" % (f.getCutoffDistance() / nanometer))
            if f.getNonbondedMethod() >= 3:
                logger.info("Ewald error tolerance : %.3e" % (f.getEwaldErrorTolerance()))
            logger.info("LJ switching function : %i" % f.getUseSwitchingFunction())
            if f.getUseSwitchingFunction():
                logger.info("LJ switching distance : %.3f nm" % (f.getSwitchingDistance() / nanometer))

# Print the sample input file here.
for line in args.record():
    print( line )

'''




