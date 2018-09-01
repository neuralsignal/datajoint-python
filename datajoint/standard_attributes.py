# -*- coding: utf-8 -*-

"""
Definitions to add to relations/schema
"""

#common attribute names

JSON = "json" #common name of json column
SUPER_COLUMN_NAMES = set((
    JSON,
    'definition',
    'description',
    'date',
    'date_modified',
    'date_created',
    'date_joined',
))
COMMON_COLUMN_NAMES = set((
    JSON,
    'definition',
    'description',
    'date',
    'date_modified',
    'date_created',
    'date_joined',
    'dict',
    'type',
    'name',
    'offset',
    'timestamps',
    'time_unit'
))

###formating tool for tables

TIMESERIES = """
    {0} = null : external #numpy array of recording
    {0}_file = null : loadstring
    timestamps = null : external #timestamps numpy array
    {0}_unit = null : varchar(63) #unit of numpy array
    time_unit = null : varchar(63) #unit of timestamps
    {0}_resolution = null : float #resolution of recording
    {0}_offset = null : float #some offset in time
    rate = null : float #rate in Hz
    {0}_date = CURRENT_TIMESTAMP : timestamp #date of recording"""
#
INTERVALSERIES = """
    {0}_starttimes : external
    {0}_stoptimes : external
    {0}_ids = null : external
    {0}_unit = null : varchar(63)
    time_unit = null : varchar(63)
    {0}_offset = null : float
    {0}_date = CURRENT_TIMESTAMP : timestamp"""

EVENTSERIES = """
    {0}_times : external #1D array with event times
    {0}_ids = null : external #multi-d array with identifiers for each step
    {0}_unit = null : varchar(63)
    time_unit = null : varchar(63)
    {0}_offset = null : float #time offset
    {0}_date = CURRENT_TIMESTAMP : timestamp"""

STRATEGY = """
    strategy_name : varchar(255)
    ---
    definition = null : varchar(4000)
    package = null : varchar(255)
    function = null : varchar(255)
    as_script = 'False' : enum('True', 'False')
    python_version = 3.6 : float
    json = null : jsonstring"""

APPROACH = """
    {name} : varchar(255)
    ---
    definition = null : varchar(4000)
    -> {strategy}
    global_settings = null : jsonstring #global settings
    entry_settings = null : jsonstring #settings taking from specific upstream columns
    date_created = CURRENT_TIMESTAMP : timestamp
    json = null : jsonstring"""
