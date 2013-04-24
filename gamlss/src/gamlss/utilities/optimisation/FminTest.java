package gamlss.utilities.optimisation;



/**
 *
 *This class tests the Fmin class.
 *
 *@author Steve Verrill
 *@version .5 --- March 25, 1998
 *
 *Modified to fit new API
 *
 *@author Andreas Maier
 *@version 0.7 --- April 27, 2010
 *
 */


public class FminTest extends Object implements SimpleOptimizableFunction{

	int id_f_to_min;
	double c,d,e;

	FminTest(int idtemp, double ctemp, double dtemp, double etemp) {

		id_f_to_min = idtemp;
		c = ctemp;
		d = dtemp;
		e = etemp;

	}

	public static void main (String args[]) {

		int another;
		int idtemp;
		double ctemp,dtemp,etemp;
		double a,b,tol,xmin;

		ctemp = dtemp = etemp = 0.0;

		another = 1;

		while (another == 1) { 
			try{

				idtemp = 1;

				if (idtemp == 1) {

					ctemp =1;
					dtemp = 1;

				} 

				FminTest fmintest = new FminTest(idtemp,ctemp,dtemp,etemp);

				a = 1;
				b =2;
				tol = 0.0001;

				SimpleFunctionOptimizer opti = new SimpleFunctionOptimizer();
				opti.setAbsoluteTolerance(tol);
				opti.setLeftEndPoint(a);
				opti.setRightEndPoint(b);
				xmin = opti.minimize(fmintest);

				System.out.print("\nThe xmin value is " + xmin + "\n");      

				another = 0;
			}catch (Exception e){
				e.printStackTrace();
			}
		}

		System.out.print("\n");

	}


	public double evaluate(double x) {

		double f;

		if (id_f_to_min == 1) {

			f = (x - c)*(x - d);

		} else if (id_f_to_min == 2) {

			f = (x - c)*(x - d)*(x - e);

		} else {

			f = Math.sin(x);

		}

		return f;         

	}


}
