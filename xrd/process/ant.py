""""""

__author__ = "Brunel Odegard"
__version__ = "0.1"


import os
import numpy as np


def fint(string):
    """Parse string to integer, allowing for strings with decimal points,
    E.G. '0.0' or '1.0' but not '1.2' """

    # first convert to float
    f = float(string)

    # error if it's not a whole number
    if not f.is_integer():
        raise ValueError("cannot interpret {} as integer".format(string))

    return int(f)

# first entry is column type.
# 0 = not implemented in ant files yet; don't include at all in the npz file
# 1 = defined per event
# 2 = defined per pulse
# 
# second entry is data type for the output array
# third entry is used to parse the values from the ant file
COL_INFO_PULSES = [
    (2 , int  , int  , "event"           ), # event number
    (1 , float, float, "timestamp"       ), # time since start of run (currently just set to event number)
    (1 , float, float, "delta_timestamp" ), # time between events (currently jusy set to 1)
    (1 , int  , int  , "np_allch"        ), # number of pulses in the event in any channel
    (1 , int  , int  , "np_ch1"          ), # number of pulses in channel 1 in the event
    (1 , int  , int  , "np_ch2"          ), # number of pulses in channel 2 in the event
    (1 , int  , int  , "np_ch3"          ), # number of pulses in channel 3 in the event
    (1 , int  , int  , "np_ch4"          ), # number of pulses in channel 4 in the event
    (2 , float, float, "pprms"           ), # RMS of the waveform containing this particular pulse, as measured from samples at the start of the waveform
    (2 , int  , int  , "channel"         ), # channel number
    (2 , int  , fint , "ipulse"          ), # pulse number within this channel (starts at 0 for first pulse)
    (2 , float, float, "tstart"          ), # time of the pulse within the waveform, in ns
    (2 , float, float, "area"            ), # area of the pulse in nVs
    (2 , float, float, "height"          ), # height of the pulse in mV
    (2 , float, float, "width"           ), # width of the pulse in ns
    (2 , float, float, "area_npe"        ), # area of the pulse calibrated to number of photoelectrons
    (2 , float, float, "area_kev"        ), # area of the pulse calibrated in keV
]

# new dirpi pulse ant file format, 30 columns
COL_INFO_DIRPIPULSES = [
    (2, int  , int  , "run"            ),
    (2, int  , int  , "evt"            ),
    (2, float, float, "tInRun"         ),
    (2, int  , int  , "tBetweenEvents" ),
    (2, int  , int  , "nPulses"        ),
    (2, int  , int  , "nCh1Pulses"     ),
    (2, int  , int  , "nCh2Pulses"     ),
    (2, float, float, "RMS1"           ),
    (2, float, float, "RMS2"           ),
    (2, int  , int  , "Ped1"           ),
    (2, int  , int  , "Ped2"           ),
    (2, int  , int  , "Area1a"         ),
    (2, int  , int  , "Area1b"         ),
    (2, int  , int  , "Area1c"         ),
    (2, int  , int  , "Area1d"         ),
    (2, int  , int  , "Area2a"         ),
    (2, int  , int  , "Area2b"         ),
    (2, int  , int  , "Area2c"         ),
    (2, int  , int  , "Area2d"         ),
    (2, int  , int  , "chan"           ),
    (2, int  , int  , "ipulse"         ),
    (2, int  , int  , "t"              ),
    (2, int  , int  , "A"              ),
    (2, int  , int  , "V"              ),
    (2, int  , int  , "width"          ),
    (2, float, float, "E"              ),
    (2, int  , int  , "dt"             ),
    (2, int  , int  , "dtOther"        ),
    (2, int  , int  , "coinc"          ),
    (2, int  , int  , "qual"           ),
]

# characters in this string will be stripped from the start and end of
# each line before it's processed
WHITESPACE = " \t\n"

# any line containint a character not in this set will be skipped.
ALLOWED_CHARS = set("0123456789 -.")




def prepare_line(line):
    # strip leading and trailing whitespace
    line = line.strip(WHITESPACE)
    return line

def load_ant_file(filepath):

    # check whether file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError("no such file: {}".format(filepath))

    # check whether file is empty
    if not os.path.getsize(filepath):
        raise Exception("empty file: {}".format(filepath))

    # open ant file and separate into lines
    with open(filepath) as f:
        lines = f.readlines()

    # return iterator of prepared lines
    return map(prepare_line, lines)



def parse_ant_file(filepath):

    # check existence and validity of ant file; return contents if valid
    lines = load_ant_file(filepath)

    # tracks which event a line in the ant file belongs to
    event = -1

    for iline, line in enumerate(lines):

        # set up lists during first line
        if iline == 0:
            columns_pulse = [[] for _ in line]
            columns_event = [[] for _ in line]

        # check line for validity
        line_chars = set(line)
        if line_chars - ALLOWED_CHARS:
            print("Invalid characters in line number {}. Line contents:".format(iline))
            print(repr(line))
            continue

        # split line into entries
        entries = line.split(" ")

        # determine whether this is a new event
        new_event = False
        if entries[0] != event:
            new_event = True
            event = entries[0]

        # populate branches
        for ie,entry in enumerate(entries):

            # parse entry (a string) into the type associated with
            # the column it's from
            # print(ie, entry, COL_INFO_PULSES[ie])
            value = COL_INFO_PULSES[ie][2](entry)

            # don't populate NYI branches
            if COL_INFO_PULSES[ie][0] == 0:
                pass

            # populate per-event branches once for each event
            if (COL_INFO_PULSES[ie][0] == 1) and new_event:
                columns_event[ie].append(value)

            # always populate per-pulse branches
            if COL_INFO_PULSES[ie][0] in (1, 2):
                columns_pulse[ie].append(value)

    # convert to dict of arrays
    arrays = {}
    for ic, (bt, bdtype, parser, bname) in enumerate(COL_INFO_PULSES):
        if bt:
            arrays[bname] = np.array(columns_pulse[ic], dtype=bdtype)
        if bt == 1:
            arrays[bname+"_event"] = np.array(columns_event[ic], dtype=bdtype)

    return arrays

def convert_ant_to_npz(filein, fileout):
    arrays = parse_ant_file(filein)
    np.savez(fileout, **arrays)




def parse_lines(lines, col_info):
    # define empty lists for population with column entries
    columns = [[] for _ in col_info]

    # parse each line, casting entries using column info
    for iline,line in enumerate(lines):
        if line.count(" ") != len(col_info) - 1:
            print("bad line {}: {}".format(iline,line))
            continue
        entries = line.split(" ")
        for ient,ent in enumerate(entries):
            columns[ient].append(col_info[ient][2](ent))

    # make and return dict of branch name to array
    return {col_info[ic][3]:np.array(c,dtype=col_info[ic][1]) for ic,c in enumerate(columns)}

def parse_pulses(lines):
    return parse_lines(lines, COL_INFO_PULSES)

def parse_dirpipulses(lines):
    return parse_lines(lines, COL_INFO_DIRPIPULSES)




def convert_file(filepath, col_info, save_npz=None):
    lines = load_ant_file(filepath)
    arrays = parse_lines(lines, col_info)
    if save_npz:
        np.savez(save_npz, **arrays)
    return arrays

def convert_pulses(filepath, save_npz=None):
    return convert_file(filepath, COL_INFO_PULSES, save_npz)

def convert_dirpipulses(filepath, save_npz=None):
    return convert_file(filepath, COL_INFO_DIRPIPULSES, save_npz)
