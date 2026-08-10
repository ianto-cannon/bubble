/*
 * run.c
 * Ianto Cannon 2026 Feb 4.  Find the maximum volume for each contact angle.
 *
 * C port of run.py.  The Python version also imports a "plot" module
 * (plot_drop_height_vs_rad, plot_graphical_abstract); those have no C
 * counterpart here and are left as commented-out stubs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bubble.h"   /* declares AdamsBashforthProfile and reorder_drop_height_vs_vol */

int main(void)
{
    /* Python's "for b in range(0):" loop is empty (range(0) -> no iterations),
       so it is intentionally omitted here. */

    /* Build RadTops = logspace(-2, 2, 1000) and use RadTop = 1 / RadTops[b]. */
    int N = 1000;
    double log_start = -2.0, log_stop = 2.0;
    for (int b = 0; b < N; ++b) {
        double t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
        double log_val = log_start + t * (log_stop - log_start);
        double radTop_entry = pow(10.0, log_val);
        double RadTop = 1.0 / radTop_entry;
        printf("%d %.6g\n", b, RadTop);
        AdamsBashforthProfile(1.0, RadTop, -1.0, NULL,
                              0.001, 0.01,
                              NULL, NULL, NULL, NULL, NULL);
    }

    reorder_drop_height_vs_vol("ang");
    reorder_drop_height_vs_vol("rad");

    /* plot_drop_height_vs_rad("loop_rad loop_ang");            */
    /* plot_graphical_abstract(...);                              */

    return 0;
}
