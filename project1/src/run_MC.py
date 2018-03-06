#!/usr/bin/env python3

"""
Command line interface to run Monte Carlo simulations using the C++ VMC codes.

For more info, run this program with the --help flag.
"""

import argparse
import sys
import os

# Some defaults in case user does not supply.
max_iterations = 1000
limits = -2, 0.6, -1, 1

parser = argparse.ArgumentParser(description=__doc__,
            epilog='You must at least supply dimensions, number of particles'
                   ' and number of MC cycles.')


parser.add_argument('--numeric', action='store_true',
                    help='Use numeric differentiation, not analytic expressions.')

parser.add_argument('--importance', action='store_true',
                    help='Use importance sampling')

parser.add_argument('--elliptic', action='store_true',
                    help='Use an elliptic trap, not a symmetrical.')

parser.add_argument('-d', '--dimensions', type=int, required=True,
                    help='Number of dimensions to use.')

parser.add_argument('-n', '--number-of-particles', type=int, required=True,
                    help='Number of particles to use.')

parser.add_argument('-mc', '--number-of-cycles', type=int, required=True,
                    help='The number of MC cycles to run.')

parser.add_argument('--alpha', type=float, default=0.5,
                    help='Value for variational parameter alpha.')

parser.add_argument('--beta', type=float, default=1,
                    help='Value for variational parameter beta.')

parser.add_argument('--time-step', type=float, default=0.001,
                    help='Time step to use in importance sampling.')

parser.add_argument('--step-length', type=float, default=1,
                    help='Step length to use in standard Metropolis.')

parser.add_argument('--omega_ho', type=float, default=1,
                    help='Strength of HO potential.')

parser.add_argument('--omega_z', type=float, default=1,
                    help='Strength of HO potential in z-dir (only for elliptical HO).')

parser.add_argument('-a', '--hard-core-diameter', type=float, default=0,
                    help='Hard-core diameter of bosons (only interaction).')

parser.add_argument('--h-derivative-step', type=float, default=0.001,
                    help='Step length to use in numerical differentiation')

parser.add_argument('-v', '--verbose', action='store_true',
                    help='Give additional output.')

parser.add_argument('-f', '--output_file', type=str, default='/dev/null',
                    help='Output file to redirect output from VMC.')





args = parser.parse_args()

# For debugging.
if args.verbose:
    print(args)

if args.verbose:
    output_command = ' | tee {}'.format(args.output_file)
else:
    output_command = ' > {}'.format(args.output_file)

command = '../build-src-Desktop-Release/run_mc.x ' + ('{} '*14).format( int(not args.numeric),
                                                                        int(args.importance),
                                                                        int(args.elliptic),
                                                                        args.dimensions,
                                                                        args.number_of_particles,
                                                                        args.number_of_cycles,
                                                                        args.alpha,
                                                                        args.beta,
                                                                        args.time_step,
                                                                        args.step_length,
                                                                        args.omega_ho,
                                                                        args.omega_z,
                                                                        args.hard_core_diameter,
                                                                        args.h_derivative_step)
to_do = command + output_command

if args.verbose:
    print(args, end='\n\n')
    print(to_do, end='\n\n')
    input('Confirm parameters and command to run: [RET]')

os.system(to_do)
