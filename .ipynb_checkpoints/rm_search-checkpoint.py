"""
parameters
---------------
- data location
- how much to average when opening data? 
- freq bands (option for all bands given and default to first one)
- how big is the chunk of time to open (& option for all time files)
- how much rfi flagging (currently have a mean and sigma threshold)
- time step in rm synthesis
- rm range?
- save info location (for script params and compute params from bash script)


questions
---------------
- consistent directory nesting format?
- consistent file format (ordering of dims etc)?
- type of data? (im doing baseband specific things rn when opening but I could assume 
    its already the full stokes data and write a separate script for opening/processing data)
- compute time?? or look at something more informative? memory usage?

"""


