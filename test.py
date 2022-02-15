#!/usr/bin/env python

#SBATCH --account=viroshaan
#SBATCH --job-name=MyJob
#SBATCH --qos=devel
#SBATCH --time=00:00:50
#SBATCH --mem-per-cpu=1
#SBATCH --ntasks=1

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load Anaconda3/2020.11
module list

## Do some work
print('hei')
