/*
 VMC for a quantum Liquid of Lennard-Jones particles
 Mass and potential constants are fixed
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h> /* necessary for the rand() function */

const int    npmax = 500;
const int    nest  = 5;
const double KB    = 1.; /* Energies in degrees Kelvin, lenghts in Angstrom */
// const double hbo2m=6.0599278;
//  hbar^2/2m (depends on mass)
const double hbo2m = 0.92756799;  //<- This value good if lengths in LJ-sigma units

double delta, rho, sigma_lj, eps_lj, l, li, l2;
double Pi, acc, vbox, b, tpb, tjf;
int    npart, nstack, ncx, ncy, ncz;

struct particle
{
    double x, y, z;
};

struct table
{
    double point[1000];
};

double       total_potential(), pair_wf(), pair_pot(), vlj(), vmod();
double       kinetic_energy(), vtail(), u();
struct table grnow(), addtable();
void         get_initial_positions(struct particle walker[npmax]);
void         advance(struct particle walker[npmax]);
void         tpsi(struct particle walker[npmax]);

struct table addtable(struct table table1, struct table table2)
{
    int          i;
    struct table temp;
    for (i = 0; i < nstack; i++)
    {
        temp.point[i] = table1.point[i] + table2.point[i];
    }
    return temp;
}

void copy_points(struct particle a[npmax], struct particle b[npmax])
{
    int i;
    for (i = 0; i < npmax; i++)
    {
        b[i].x = a[i].x;
        b[i].y = a[i].y;
        b[i].z = a[i].z;
    }
}

int main()
{
    struct particle walker[npmax];
    struct table    gr;
    double          vout;
    double          dr, l22;
    double          dest[nest], estave[nest], estave2[nest];
    double estsum[nest], estsum2[nest], estnow[nest], estvar[nest]; /* estimators */
    int    idest[nest];
    double vbulk, accum;
    double gr_norm, r, norm, rout, rin;
    float  delta_in, rho_in, b_in;
    int    nstep1, nstep2;
    int    i, j;
    FILE*  out;

    out = fopen("gr.out", "w+");
    Pi  = 4. * atan(1.);

    printf("Quantum Lennard-Jones fluid in 3D: \n");

    /******************/
    /* Parameters of the LJ potential. These values are for atomic 4He,
       but they could be set = 1 for a generic system
    */
    sigma_lj = 1.000;
    eps_lj   = 10.22;
    /*******************/
    printf("Input particle number: ");
    scanf("%i", &npart);
    printf("Number of fcc cells along x, y, and z directions: ");
    scanf("%i %i %i", &ncx, &ncy, &ncz);
    printf("Number density of the system:");
    scanf("%f", &rho_in);
    rho = rho_in;
    printf("Value of the variational parameter b: ");
    scanf("%f", &b_in);
    b = b_in;
    printf("Step for the Metropolis algorithm: ");
    scanf("%f", &delta_in);
    delta = delta_in;
    printf("Number of equilibration steps : ");
    scanf("%i", &nstep1);
    printf("Number of steps: ");
    scanf("%i", &nstep2);
    printf("Number of points to compute g(r): ");
    scanf("%i", &nstack);

    /* This part computes quantities that are proper for
       a simulation in a periodic box */

    l  = pow((double) npart / rho, 1. / 3.);
    li = 1. / l;
    l2 = 0.5 * l;

    printf("SIMULATION CELL SIDE: %10.8f \n", l);
    vbox  = 0.;
    l22   = l2 * l2 - 1.e-10;
    vbox  = vlj(l22);
    vout  = vtail(l);
    vout  = 0.;
    vbulk = 0.5 * vbox * (Pi * npart / 6. - 1);
    printf("TAIL CORRECTION: %10.5e\n", vout);
    printf("POTENTIAL AT L/2: %10.5e %10.5e\n", vbox, vlj(l2 * l2));
    printf("INTERNAL CORRECTION: %10.5e\n", vbulk);

    /* Initialize the random number generator */

    srand(1);

    /* Call a subroutine initializing particle positions */

    get_initial_positions(walker);

    /* Zero out the estimates cumulants */
    accum = 0.;
    /************************************************/
    /*   BEGINNING OF THE MAIN LOOP                 */
    /*      EQUILIRATION STAGE                      */
    /************************************************/
    for (i = 0; i < nstep1; i++)
    {
        advance(walker); /* <- Routine to "advance" the walkers (Metropolis step) */
        accum = accum + acc;
        if ((i + 1) % 100 == 0)
            printf("Step: %6i  Acc: %10.5f \n", i + 1, accum / (npart * (i + 1)));
    }
    printf("\n End of the equilibration stage \n");
    /************************************************/
    /*   BEGINNING OF THE MAIN LOOP                 */
    /*      COMPUTATION STAGE                       */
    /************************************************/
    for (j = 0; j < nest; j++)
    {
        estsum[j]  = 0.;
        estsum2[j] = 0.;
    }
    printf(
        "Step    Enow      Eave    Eerr   TPBave    TPBave     TJFave    TJFave    "
        "Vnow     Vave      Acc.\n");
    for (i = 0; i < 201; i++)
    {
        gr.point[i] = 0;
    }
    for (i = 0; i < nstep2; i++)
    {
        advance(walker);
        tpsi(walker); /* <- this routine computes the kinetic energy */
                      /* In this block estimates are evaluated and cumulated */
        estnow[0] = total_potential(walker) / (double) npart + vout + vbulk;
        estnow[1] = estnow[0] + tpb / (double) npart;
        estnow[3] = tpb / (double) npart;
        estnow[4] = tjf / (double) npart;
        estnow[2] = acc / (double) npart;
        gr        = addtable(gr, grnow(walker));
        for (j = 0; j < nest; j++)
        {
            estsum[j] = estsum[j] + estnow[j]; /* <- Sum up the estimators */
            estsum2[j]
                = estsum2[j] + estnow[j] * estnow[j]; /* <- Sum up estimators squared
                                                         (for computing the errors) */
            estave[j] = estsum[j] / (i + 1); /* <- Compute the running average  */
            estave2[j]
                = estsum2[j] / (i + 1); /* <- Compute the average of the squares */
            estvar[j] = fabs(estave[j] * estave[j]
                             - estave2[j]); /* <- Build the estimator of the variance */
            dest[j]   = sqrt(
                estvar[j]
                / (double) (i + 1)); /* <- Current estimate of the satistical error */
            idest[j] = dest[j] * 1.e5; /* <- This has just a cosmetic purpose for a
                                          pretty printing of errors */
        }
        if ((i + 1) % 100 == 0)
            printf(
                "%6i %10.5f %10.5f (%5i)  %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f "
                "%10.5f\n",
                (i + 1),
                estnow[1],
                estave[1],
                idest[1],
                estnow[3],
                estave[3],
                estnow[4],
                estave[4],
                estnow[0],
                estave[0],
                estave[2]);
    }
    /**********************************************************/
    /*      END OF THE MAIN LOOP                              */
    /**********************************************************/
    /* Final printing */
    for (j = 0; j < nest; j++)
    {
        estave[j]  = estsum[j] / nstep2;
        estave2[j] = estsum2[j] / nstep2;
        dest[j]    = sqrt(fabs(estave[j] * estave[j] - estave2[j]) / (double) nstep2);
    }
    /* UNITS CHANGE! energies in epsilon units, multiplied by N */
    double cfact;
    cfact = npart / eps_lj;
    printf("Average potential energy: %10.8f +-  %10.8f\n",
           estave[0] * cfact,
           dest[0] * cfact);
    printf("Average PB kinetic energy: %10.8f +-  %10.8f\n",
           estave[3] * cfact,
           dest[3] * cfact);
    printf("Average JF kinetic energy: %10.8f +-  %10.8f\n",
           estave[4] * cfact,
           dest[4] * cfact);
    printf("Average total energy: %10.8f +-  %10.8f\n",
           estave[1] * cfact,
           dest[1] * cfact);
    printf("Variance of the total energy: %10.8f\n", estvar[4] * cfact);

    /* This block computes the normalization of the pdf and prints it out */
    for (i = 0; i < nstack; i++)
    {
        dr      = l * 0.5 / (double) (nstack + 1); /* g(r) */
        r       = (i + 0.5) * dr;
        rin     = i * dr;
        rout    = (i + 1) * dr;
        norm    = 4. / 3. * Pi * (rout * rout * rout - rin * rin * rin) * rho;
        gr_norm = (double) gr.point[i] / ((double) nstep2 * norm * npart);
        fprintf(out, "%10.5f %10.5f\n", r, gr_norm);
    }
    fclose(out);
}

/* SUBROUTINE TO INITIALIZE PARTICLE POSITIONS
   builds the initial configuration placing particles on
   a lattice (convenient for bulk systems)
*/

void get_initial_positions(struct particle walker[npmax])
{
    double basisx[4] = {0., 0.5, 0.0, 0.5};
    double basisy[4] = {0., 0.5, 0.5, 0.0};
    double basisz[4] = {0., 0.0, 0.5, 0.5};

    int    i, j, k, ll;
    double side, xd, yd, zd;
    int    index;

    side = pow(4. / rho, 1. / 3.);

    printf("elementary cell side: %10.5f\n", side);

    index = 0;
    for (i = 0; i < ncx; i++)
    {
        for (j = 0; j < ncy; j++)
        {
            for (k = 0; k < ncz; k++)
            {
                for (ll = 0; ll < 4; ll++)
                {
                    walker[index].x = ((double) i + basisx[ll]) * side;
                    walker[index].y = ((double) j + basisy[ll]) * side;
                    walker[index].z = ((double) k + basisz[ll]) * side;
                    index++;
                }
            }
        }
    }
    if (index != npart)
    {
        printf("ERROR: INCONSISTENT NUMBER OF PARTICLES  %5i  %5i", index, npart);
        abort();
    };
    for (j = 0; j < npart; j++)
    {
        xd          = walker[j].x;
        walker[j].x = xd - l * rint(xd * li);
        yd          = walker[j].y;
        walker[j].y = yd - l * rint(yd * li);
        zd          = walker[j].z;
        walker[j].z = zd - l * rint(zd * li);
        printf("%10.5f  %10.5f   %10.5f\n", walker[j].x, walker[j].y, walker[j].z);
    }
}

/* SUBROUTINE PERFORMING THE METROPOLIS STEP */

void advance(struct particle walker[npmax])

{
    struct particle new;
    int    j;
    double dx, dy, dz, arg;
    double uo, un, p, csi;
    acc = 0.;
    /* Here we loop over the particle index.
       Each particle is displaced and then tested for
       acceptance-rejection of the move.
       The step might also be performed displacing all
       particles first and testing at the end.
       From the point of view of the generation of the
       Markov chain the two processes are equivalent.
       Autocorrelations might be substantially different.
    */
    for (j = 0; j < npart; j++)
    {
        uo = pair_wf(
            j,
            walker[j].x,
            walker[j].y,
            walker[j].z,
            walker); /* Here we compute the contributions to the wavefunction
                        relative to the particle j that is going to be displaced */
        dx = delta
             * (0.5
                - (double) rand()
                      / (double) RAND_MAX); /* <- generate a random displacement in a
                                               cube of side delta */
        new.x
            = walker[j].x + dx; /* <- add the displacement to the particle coordinate */
        new.x
            = new.x
              - l* rint(new.x* li); /* <- this line imposes periodic boundary conditions
                                       (coordinates are set back in the simulation box.
                                       It is not needed if particles are confined! */
        dy    = delta * (0.5 - (double) rand() / (double) RAND_MAX);
        new.y = walker[j].y + dy;
        new.y = new.y - l* rint(new.y* li);

        dz    = delta * (0.5 - (double) rand() / (double) RAND_MAX);
        new.z = walker[j].z + dz;
        new.z = new.z - l* rint(new.z* li);

        un = pair_wf(j,
                     new.x,
                     new.y,
                     new.z,
                     walker); /* Here we compute the contributions to the wavefunction
                               relative to the particle j but this time after it has
                               been displaced */

        arg = un - uo;
        p = exp(-arg); /* This is the ratio between the wavefunctions squared before and
                          after the attempted move i.e. the acceptance probability */

        csi = (double) rand() / (double) RAND_MAX; /*<- extract a random number between
                                                      0 and 1 to test acceptance */

        if (p > csi) /* if p>csi the move is accepted, and the position of the particle
                        is updated.
                        Notice that id p>1, the move is going to be accepted anyway */
        {
            acc         = acc + 1.;
            walker[j].x = new.x;
            walker[j].y = new.y;
            walker[j].z = new.z;
        };
        /*      printf("arg: %10.5f p: %10.5f acc: %10.5f \n",arg,p,acc); */
    };
}

/* SUBROUTINE TO COMPUTE THE POTENTIAL ENERGY */

double total_potential(struct particle walker[npmax])
{
    int    i, j;
    double dx, dy, dz, rr, v;
    v = 0.;
    /* This is the double loop over the particles to build the
       sum over i<j. Notice that in this way the pairs are
       counted only once! */
    for (i = 1; i < npart; i++)
    {
        for (j = 0; j < i; j++)
        {
            dx = walker[i].x - walker[j].x;
            dx = dx - l * rint(dx * li); /* <- This line imposes periodic boundary
                                            conditions on the distances. It is not
                                            needed for a confined system */
            dy = walker[i].y - walker[j].y;
            dy = dy - l * rint(dy * li);
            dz = walker[i].z - walker[j].z;
            dz = dz - l * rint(dz * li);
            rr = dx * dx + dy * dy + dz * dz;
            v  = v + vlj(rr); /* The function vlj computes the value of the potential */
        };
    };
    return v;
}

/* SUBROUTINE TO COMPUTE THE CONTRIBUTIONS TO THE WAVEFUNCTION FROM PARTICLE i */

double pair_wf(int i, double x, double y, double z, struct particle walker[npmax])
{
    int    j;
    double dx, dy, dz, rr, v;

    v = 0.;
    /* In this two loops all the distances between particle i and all the other
       particles are computed */
    for (j = 0; j < i; j++)
    {
        dx = x - walker[j].x;
        dx = dx - l * rint(dx * li); /* periodic boundary conditions (see above) */
        dy = y - walker[j].y;
        dy = dy - l * rint(dy * li);
        dz = z - walker[j].z;
        dz = dz - l * rint(dz * li);
        rr = dx * dx + dy * dy + dz * dz;
        v = v + u(rr); /* The function u computes the value of the wavefunction square*/
    };
    for (j = i + 1; j < npart; j++)
    {
        dx = x - walker[j].x;
        dx = dx - l * rint(dx * li);
        dy = y - walker[j].y;
        dy = dy - l * rint(dy * li);
        dz = z - walker[j].z;
        dz = dz - l * rint(dz * li);
        rr = dx * dx + dy * dy + dz * dz;
        v  = v + u(rr);
    };
    return v;
}

/* SUBROUTINE TO COMPUTE THE LJ POTENTIAL */

double vlj(double r2)
{
    double r2i, r6i, r12i, v, l22;
    double rcore;

    l22 = l2 * l2; /* This line computes the square of the cell side */
    if (r2 > l22) /* If the distance of two particles exceeds half of the simulation box
                     side the contribution of the pair is zero. This is strictly
                     necessary only for periodic systems. */
    {
        v = 0;
        return v;
    };

    rcore = 0.3 * sigma_lj; /* In order to avoid overflows the potential is set equal
                               to a constant if the distance of a pair of particles is
                               less that 1/3 sigma. The probability of such event is low
                               (the wavefunction is almost 0 there), and this has a
                               very limited impact on the final result */

    if (r2 < rcore * rcore)
        r2 = rcore * rcore;
    r2i  = sigma_lj * sigma_lj / r2;
    r6i  = r2i * r2i * r2i;
    r12i = r6i * r6i;
    v    = 4. * eps_lj * (r12i - r6i) - vbox;
    return v;
}

/* SUBROUTINE TO COMPUTE THE WAVEFUNCTION */
/* All the comments made for the potential
   (subroutine above), apply here   */

double u(double r2)
{
    double ri, r2i, r5i, v, l22;
    double rcore;

    l22 = l2 * l2;
    if (r2 > l22)
    {
        v = 0;
        return v;
    };

    rcore = 0.3 * sigma_lj;

    if (r2 < rcore * rcore)
        r2 = rcore * rcore;
    ri  = b / sqrt(r2);
    r2i = b * b / r2;
    r5i = r2i * r2i * ri;
    v   = r5i; /* IMPORTANT!!!! Here we compute b/r^5 WITHOUT THE FACTOR 1/2!
                  This is already the square of the wavefunction */
    return v;
}

/* SUBROUTINE TO COMPUTE THE KINETIC ENERGY */

void tpsi(struct particle walker[npmax])
{
    struct particle dpsi[npmax];
    double          d2psi;
    int             i, j;
    double          dx, dy, dz, r2, v, rcore;
    double          l22, dux, duy, duz, r2i, r6i, ri;
    /* zero out the variables containing the derivative of the
       pseudopotential of the pair wavefunction (v), the gradient
       (dpsi, notice that it has 3N components), and the
       laplacian (d2psi) */
    v = 0.;
    for (i = 0; i < npart; i++)
    {
        dpsi[i].x = 0.;
        dpsi[i].y = 0.;
        dpsi[i].z = 0.;
    };
    d2psi = 0.;
    /* We start here by computing grad(Psi)/Psi */
    for (i = 1; i < npart; i++) /*  double loop over the pairs (see above)*/
    {
        for (j = 0; j < i; j++)
        {
            dx = walker[i].x - walker[j].x; /* x component of the particle distance */
            dx = dx
                 - l * rint(dx * li); /*<- periodic boundary conditions (see above) */

            dy = walker[i].y - walker[j].y;
            dy = dy - l * rint(dy * li);

            dz = walker[i].z - walker[j].z;
            dz = dz - l * rint(dz * li);

            r2  = dx * dx + dy * dy + dz * dz; /* total distance squared */
            l22 = l2 * l2;
            if (r2 < l22) /* <- Check if the distance is less than half of the side of
                             the simulation box. Strictly needed only for periodic
                             systems */
            {
                rcore = 0.3 * sigma_lj; /* Set a minimal distance to avoid overflow (see
                                           above) */
                if (r2 < rcore * rcore)
                    r2 = rcore * rcore;
                r2i = b * b / r2;
                ri  = 1. / (b * sqrt(r2));
                r6i = r2i * r2i * r2i;
                v   = -5. * r6i * ri; /* <- This is u'(r_ij) */
                dux = v * dx;         /* These lines compute
                                         \vec(f_ij)=-u'(r_ij)/r_ij*\vec(r_ij) */
                duy = v * dy;
                duz = v * dz;
                dpsi[i].x
                    = dpsi[i].x + dux; /* <- Here the gradient is computed. It should be
                                             \sum_{j<>i} -u'(r_ij)/r_ij*\vec(r_ij),
                                             but we can exploit the fact that
                                             grad(u)i(r_ij) = -gradi(u)(r_ji) and
                                             reuse the contribution of the pair both for
                                             the gradient with respect to r_i and with
                                             respect to r_j */
                dpsi[i].y = dpsi[i].y + duy;
                dpsi[i].z = dpsi[i].z + duz;
                dpsi[j].x = dpsi[j].x
                            - dux; /* <- this is how grad(u)_ij = -gradu_ji is used */
                dpsi[j].y = dpsi[j].y - duy;
                dpsi[j].z = dpsi[j].z - duz;

                d2psi = d2psi + 20. * r6i * ri; /* <- computation of the laplacian of
                                                       the pseudopotential (very simple,
                                                   it is a scalar) */
            }
        };
    };
    tpb = 0.;
    tjf = 0.;
    /* Once we have computed the gradient of the pseudopotential, we
       need to square it to compute the laplacian of the wavefunction */
    for (i = 0; i < npart; i++)
    {
        tpb = tpb + 0.25 * (dpsi[i].x * dpsi[i].x);
        tpb = tpb + 0.25 * (dpsi[i].y * dpsi[i].y);
        tpb = tpb + 0.25 * (dpsi[i].z * dpsi[i].z);
    };
    tpb = hbo2m * (d2psi - tpb); /* Finally, here we compute the whole laplacian and
                                    multiply by -\hbar^2/2m */
    tjf = 0.5 * hbo2m * d2psi; /* This line gives an alternate expression of the kinetic
                                  energy obtained integrating the laplacian by parts,
                                  summing the result to the laplacian itself and
                                  dividing by 2 (Jackson-Feenberg identity). */
}

/* SUBROUTINE TO COMPUTE THE PAIR DISTRIBUTION FUNCTION */
/* In this subroutine an histogram is built of all the
   particle-particle distances */

struct table grnow(struct particle walker[npmax])
{
    int          i, j;
    double       dx, dy, dz, rr;
    double       r, dh;
    int          idx;
    struct table gg;
    dh = l2 / (double) (nstack + 1);

    for (i = 0; i < 201; i++)
        gg.point[i] = 0;
    for (i = 0; i < npart; i++)
    {
        for (j = 0; j < i; j++)
        {
            dx = walker[i].x - walker[j].x;
            dx = dx - l * rint(dx * li);
            dy = walker[i].y - walker[j].y;
            dy = dy - l * rint(dy * li);
            dz = walker[i].z - walker[j].z;
            dz = dz - l * rint(dz * li);
            rr = dx * dx + dy * dy + dz * dz;
            r  = sqrt(rr);
            if (r < l2)
            {
                idx           = (int) (r / dh);
                gg.point[idx] = gg.point[idx] + 2;
            };
        };
    };
    return gg;
}

/* SUBROUTINE TO EVALUATE TAIL CORRECTIONS */
/* Needed in periodic systems only. These
   expressions account for the contributions
   to the potential energy that are not included due
   to the cut of distances at half of the simulation
   box side. The assumption is that beyond that
   distance the distribution (density) of particles
   is strictly uniform */

double vtail(double l)
{
    double aux, aux3, aux9, v, sigma3;

    aux    = 2. * sigma_lj * li;
    aux3   = aux * aux * aux;
    aux9   = aux3 * aux3 * aux3;
    sigma3 = sigma_lj * sigma_lj * sigma_lj;
    v      = 8. / 9. * Pi * eps_lj * rho * sigma3 * (aux9 - 3. * aux3);
    return v;
}
