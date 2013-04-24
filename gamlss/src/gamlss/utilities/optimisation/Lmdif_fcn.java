package gamlss.utilities.optimisation;
public interface Lmdif_fcn {

   void fcn(int m, int n, double x[], double fvec[],
            int iflag[]);

}
