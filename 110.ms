# Enter title and control flags
_sf _file _cor _new _abs

# Atomic coordinate file
dat\Si110_20x14x1.dat
# HV tiltx tilty
300 0.0 0.0

# meshx meshy
1024 1024

# Define surface layer t0 t1 n
0.000000 0.000000 0

# Define bulk layer t0 t1 n layer
-0.000001 7.6792 4 50

# Extinction rule 1 1 default
1 1

# 1 to calculate potential in real space
1

# Specify the atom and the Debye-Waller factor
1
Si
0.4668

# Mean absorption
0.10

# Objective aperture
7.00

# Output
160 -1

# Pendulusung output
si110.pl
1 0 0

# STEM control
STEM-DIF

# Probe parameters
kmax 0.9
df_nm 0.0
c3_mm 0.0
c5_mm 0.0

# Probe position
probe 0.5698 0.5457

# End of probe parameter input
end

# Scan area and points in x and y directions
0.050000 0.0714285 2 2 

