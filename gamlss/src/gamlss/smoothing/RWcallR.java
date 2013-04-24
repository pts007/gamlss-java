package gamlss.smoothing;

import java.util.HashMap;
import java.util.Hashtable;

import gamlss.algorithm.AdditiveFit;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.utilities.Controls;
import gamlss.utilities.MatrixFunctions;
import gamlss.utilities.oi.ConnectionToR;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.util.FastMath;


/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public class RWcallR implements GAMLSSSmoother {
	/** */
	private ArrayRealVector delta; 
	/** b equals to fitted values lagged by Controls.RW_ORDER. */
	private ArrayRealVector b;
	/** Value of Objective function. */
	private double valueOfQ;
	/** */
	private ArrayRealVector se;
	/** Identity matrix size(y) over size(y). */
	private BlockRealMatrix eM;
	/** dM equals to eM lagged by Controls.RW_ORDER. */
	private BlockRealMatrix dM;
	/** Diagonal matrix with the vector of weights at the main diagonal. */
	private BlockRealMatrix wM;
	/** gM = dM*transpose(dM). */
	private BlockRealMatrix gM;
	/** length of the response variable vector. */
	private int n;
	/** Fitted Values. */
	private ArrayRealVector fv;
	/** Object of non-linear optimisation objective function, 
	 * for the case when neither logsig2e and logsig2b is fixed.  */
	private NonLinObjFunction nonLinObj;
	/** Object of non-linear optimisation objective function, 
	 * for the case when logsig2e is fixed.  */
	private NonLinObjFunction1 nonLinObj1;
	/** Object of non-linear optimisation objective function, 
	 * for the case when logsig2b is fixed.  */
	private NonLinObjFunction2 nonLinObj2;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Hashtable to store sigmas. */
	private Hashtable<Integer, Double[]> sigmasHash;
	/** Object of a class for Maximum number of iterations 
	 * performed by an (iterative) algorithm. */
	private MaxIter maxiter;
	/** Object of a class for Maximum number of evaluations 
	 * of the function to be optimized. */
	private MaxEval maxeval;
	/** Object of a class for Starting point (first guess) 
	 * of the optimization procedure. */
	private InitialGuess initguess;
	/** Object of a class for Simple optimization constraints: 
	 * lower and upper bounds. */
	private SimpleBounds bounds;
	/** Object of a class for Scalar function to be optimized. */
	private ObjectiveFunction obj;
	/** Object of DecompositionSolver class.*/
	private DecompositionSolver solver;
	/**Object of a class for Powell's BOBYQA optimisation algorithm. */
	private BOBYQAOptimizer optimizer;
	/** Response variable. */
	private ArrayRealVector y;
	/**  Weights. */
	private ArrayRealVector w;
	/**  Fitting distribution parameter. */
	private int whichDistParameter;
	/**  Object of ConnectionToR class. */
	private ConnectionToR rConnection;
	
	
	/**
	 * Constructor of Random Walk Class.
	 */
	public RWcallR(GAMLSSFamilyDistribution distr, ConnectionToR rConnect, 
							HashMap<Integer, BlockRealMatrix> smoothMatrices) {		
		//if(order < 0) {
		if (Controls.RW_ORDER < 0) { 
			Controls.RW_ORDER = 1;
			System.err.println("the value of order supplied is zero"
				+ " or negative the default value of 1 was used instead");
		}
				
		//if(sig2e < 0)
		if (Controls.RW_SIG2E < 0) { 
			Controls.RW_SIG2E = 1.0;
			System.err.println("the value of sig2e supplied is zero"
					+ " or negative the default value of 1 was used instead");
		}
				
		//if(sig2b < 0)
		if (Controls.RW_SIG2B < 0) { 
			Controls.RW_SIG2B = 1.0;
			System.err.println("the value of sig2b supplied is zero"
					+ " or negative the default value of 1 was used instead");
		}
		
		delta = new ArrayRealVector(Controls.RW_DELTA, false);
		
		// if(any(delta <= 0)||length(delta)!=2)
		if (delta.getMinValue() < 0 || delta.getDimension() != 2) {
			
			//delta <- c(0.01, 0.01)
			delta = MatrixFunctions.repV(0.01, 2);
			System.err.println("delta should be positive and of length 2, "
							 + "deltat is set at default values c(0.01,0.01)");
		}
				
		//if(length(shift)!=2)
		if (Controls.RW_SHIFT.length != 2) {
			
			//shift <- c(0, 0)
			Controls.RW_SHIFT = new double[2];
			System.err.println("shift  length should be 2, is it set" 
												 + "at default values c(0,0)");
		} 
	
		sigmasHash = new Hashtable<Integer, Double[]>();
		for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
			if (smoothMatrices.get(i) != null) { 
				sigmasHash.put(i, new Double[] {Controls.RW_SIG2E, Controls.RW_SIG2B});
			}
		}
		optimizer = new BOBYQAOptimizer(Controls.BOBYQA_INTERPOL_POINTS);
		
		
		if (Controls.SIG2E_FIX) {
			if (Controls.SIG2B_FIX) {
				nonLinObj = null;
			} else {
				nonLinObj1 = new NonLinObjFunction1();
			} 
		} else {
			if (Controls.SIG2B_FIX) {
				nonLinObj2 = new NonLinObjFunction2();
			} else {
				nonLinObj = new NonLinObjFunction();
			}
		}
		
		rConnect.voidEval("library(spam)");
		if (!Controls.JAVA_OPTIMIZATION) {
			rConnect.voidEval("Qf <- function(par) " +
				"{fv <- solve((exp(-par[1])*W)" +
				" + (exp(-par[2])*G)," +
				" (exp(-par[1])*W)%*%as.vector(y));" +
				" b <- diff(fv, differences=order);" +
				" DT <-  determinant((exp(-par[1])*W) + " +
				"(exp(-par[2])*G))$modulus; " +
				"f <- -(N/2)*log(2*pi*exp(par[1]))" +
				" +.5*sum(log(weights))" +
				"-sum(weights*(y-fv)^2)/(2*exp(par[1])) " +
				"+(N/2)*log(2*pi)" +
				"-((N-order)/2)*log(2*pi*exp(par[2]))" +
				"-sum(b^2)/(2*exp(par[2])) -.5*DT;" +
				" attributes(f) <- NULL; -f}");
		}
		this.rConnection = rConnect;

	}
	
	/**
	 * The main fitting method of Random Walk smoother.
	 * @param additiveFit - object of AdditiveFit class
	 * @return residuals
	 */
	public ArrayRealVector solve(AdditiveFit additiveFit) {
	
		y = additiveFit.getZ();
		w = additiveFit.getW();
		whichDistParameter = additiveFit.getWhichDistParameter();
		
        //if (any(is.na(y)))
		for (int i = 0; i < y.getDimension(); i++) {
			if (Double.isNaN(y.getEntry(i))) {
				
		        //weights[is.na(y)] <- 0
				w.setEntry(i, 0.0);
				
		        //y[is.na(y)] <- 0
				y.setEntry(i, 0.0);
			}
		}
		
        //N <- length(y)
		n = y.getDimension();

        //W <- diag.spam(x=weights)  # weights
		rConnection.assingVar("weights", w.getDataRef());
		rConnection.voidEval("W <- diag.spam(x=weights)");
		
        //E <- diag.spam(N)
		rConnection.assingVar("N", new double[]{n});
		rConnection.voidEval("E <- diag.spam(N)");
					
        //D <- diff(E, diff = order) # order 
		rConnection.assingVar("order", new double[]{Controls.RW_ORDER});
		rConnection.voidEval("D <- diff(E, diff = order)");
				
        //G <- t(D)%*%D
		rConnection.voidEval("G <- t(D)%*%D");
				
        //logsig2e <- log(sig2e) # getting logs
		double logsig2e = FastMath.log(sigmasHash.get(whichDistParameter)[0]);
		
        //logsig2b <- log(sig2b)
		double logsig2b = FastMath.log(sigmasHash.get(whichDistParameter)[1]);

		rConnection.assingVar("logsig2e", new double[]{logsig2e});
		rConnection.assingVar("logsig2b", new double[]{logsig2b});
		rConnection.assingVar("y", y.toArray());
		
		fv = new ArrayRealVector(rConnection.runEvalDoubles(
		    "fv <- solve(exp(-logsig2e)*W + exp(-logsig2b)*G,(exp(-logsig2e)*W)%*% as.vector(y))"));
		
        //if (sig2e.fix==FALSE && sig2b.fix==FALSE) # both estimated{
		if (Controls.SIG2E_FIX) {
			if (Controls.SIG2B_FIX) {
				
                //out <- list()	
                //par <- c(logsig2e, logsig2b)
                //out$par <- par
                //value.of.Q <- -Qf(par)
				valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
				
                //se <- NULL 
				se = null;
				
			} else {
				
                //par <- log(sum((D%*%fv)^2)/N)	
				tempV = new ArrayRealVector(rConnection.runEvalDoubles(
						"tempV <- D%*%fv"));

				logsig2b = FastMath.log(MatrixFunctions.sumV(
												tempV.ebeMultiply(tempV)) / n);
							
                //Qf1 <- function(par)  Qf(c(logsig2e, par)) 
                //out<-nlminb(start = par,objective = Qf1,
				//lower = c(-20), upper = c(20))
				
				nonLinObj1.setResponse(y);
				nonLinObj1.setWeights(w);
				nonLinObj1.setLogsig2e(logsig2e);
				nonLinObj1.setrConnection(rConnection);
				   
				maxiter = new MaxIter(Controls.BOBYQA_MAX_ITER);
				maxeval = new MaxEval(Controls.BOBYQA_MAX_EVAL);
				initguess = new InitialGuess(new double[] {logsig2b, logsig2b});
				bounds = new SimpleBounds(
						   new double[] {-Double.MAX_VALUE, -Double.MAX_VALUE},
						   new double[] {Double.MAX_VALUE, Double.MAX_VALUE});
				obj = new ObjectiveFunction(nonLinObj1);
				PointValuePair values = optimizer.optimize(maxiter, 
														   maxeval, 
														   obj, 
														   initguess, 
														   bounds, 
														   GoalType.MINIMIZE);
				
				logsig2b = values.getPoint()[0];
				
				if (logsig2b > 20.0){
					logsig2b = 20.0;
					valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
				} else if (logsig2b < -20.0){
					logsig2b = -20.0;
					valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
				} else {
					valueOfQ = -values.getValue();
				}
				
				//out$hessian <- optimHess(out$par, Qf1)
				//value.of.Q <- -out$objective
				//shes <- 1/out$hessian
                //se1 <- ifelse(shes>0, sqrt(shes), NA) 
                //par <- c(logsig2e, out$par) 
				//names(par) <- c("logsig2e", "logsig2b")  
				///out$par <- par    
				//se <- c(NA, se1)		
//System.out.println(logsig2e+"   "+logsig2b +"   "+qF(new double[]{logsig2e, logsig2b}, y, w));
			} 
		} else {
			if (Controls.SIG2B_FIX) {
				
	               	//par <- log(sum(weights*(y-fv)^2)/N)   
					tempV = y.subtract(fv);
					logsig2e = FastMath.log(MatrixFunctions.sumV(
								 tempV.ebeMultiply(tempV).ebeMultiply(w)) / n);
				
					//Qf2 <- function(par)  Qf(c(par, logsig2b))  	
					//out <- nlminb(start = par, objective = Qf2, 
					//lower = c(-20),  upper = c(20))
					nonLinObj2.setResponse(y);
					nonLinObj2.setWeights(w);
					nonLinObj2.setLogsig2b(logsig2b);
					nonLinObj2.setrConnection(rConnection);
					
					   
					maxiter = new MaxIter(Controls.BOBYQA_MAX_ITER);
					maxeval = new MaxEval(Controls.BOBYQA_MAX_EVAL);
					initguess = new InitialGuess(new double[] {logsig2e, logsig2e});
					bounds = new SimpleBounds(
							new double[] {-Double.MAX_VALUE, -Double.MAX_VALUE},
							new double[] {Double.MAX_VALUE, Double.MAX_VALUE});
					obj = new ObjectiveFunction(nonLinObj2);
					PointValuePair values = optimizer.optimize(maxiter, 
															   maxeval, 
															   obj, 
															   initguess, 
															   bounds, 
															   GoalType.MINIMIZE);
					
					logsig2e = values.getPoint()[0];
					
					//out$hessian <- optimHess(out$par, Qf2)
					//value.of.Q <- -out$objective
					if (logsig2e > 20.0){
						logsig2e = 20.0;
						valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
					} else if (logsig2e < -20.0){
						logsig2e = -20.0;
						valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
					} else {
						valueOfQ = -values.getValue();
					}
				
					
	               //shes <- 1/out$hessian
	               //se1 <- ifelse(shes>0, sqrt(shes), NA) 
	               //par <- c( out$par, logsig2b) 
				   //names(par) <- c("logsig2e", "logsig2b")  
				   //out$par <- par    
	               //se <- c(se1, NA)
//System.out.println(logsig2e+"   "+logsig2b +"   "+qF(new double[]{logsig2e, logsig2b}, y, w));					
				
			} else {

			    //par <- c(logsig2e <- log(sum(weights*(y-fv)^2)/N), 
				//logsig2b <-log(sum((D%*%fv)^2)/N))
				tempV = y.subtract(fv);
				logsig2e = FastMath.log(MatrixFunctions.sumV(
								 w.ebeMultiply(tempV).ebeMultiply(tempV)) / n);
				
				//tempV = (ArrayRealVector) dM.operate(fv);
				tempV = new ArrayRealVector(rConnection.runEvalDoubles(
															"tempV <- D%*%fv"));

				logsig2b = FastMath.log(MatrixFunctions.sumV(
												tempV.ebeMultiply(tempV)) / n);
				
				if (Controls.JAVA_OPTIMIZATION) {
				nonLinObj.setResponse(y);
				nonLinObj.setWeights(w);
				nonLinObj.setrConnection(rConnection);
		
				 //out <- nlminb(start = par, objective = Qf, 
				 //lower = c(-20, -20),  upper = c(20, 20))
				 maxiter = new MaxIter(Controls.BOBYQA_MAX_ITER);
				 maxeval = new MaxEval(Controls.BOBYQA_MAX_EVAL);
				 initguess = new InitialGuess(new double[] {logsig2e, logsig2b});
				 bounds = new SimpleBounds(new double[] {-40.0, -40.0},//{-Double.MAX_VALUE, -Double.MAX_VALUE}, 
						 				   new double[] {40.0, 40.0});//{Double.MAX_VALUE, Double.MAX_VALUE});
				 obj = new ObjectiveFunction(nonLinObj);

				PointValuePair values = optimizer.optimize(maxiter, 
														   maxeval, 
														   obj, 
														   initguess, 
														   bounds, 
														   GoalType.MINIMIZE);		
				
			
				
				logsig2e = values.getPoint()[0];
				logsig2b = values.getPoint()[1];
				
				boolean switcher = false;
				
				if (logsig2e > 20.0){
					logsig2e = 20.0;
					switcher = true;
				} else if (logsig2e < -20.0){
					logsig2e = -20.0;
					switcher = true;
				} 
				
				if (logsig2b > 20.0){
					logsig2b = 20.0;
					switcher = true;
				} else if (logsig2b < -20.0){
					logsig2b = -20.0;
					switcher = true;
				} 
				
				if (switcher) {
					valueOfQ = -qF(new double[]{logsig2e, logsig2b}, y, w, rConnection);
				} else {
					valueOfQ = -values.getValue();
				}
				
//System.out.println(logsig2e+"   "+logsig2b +"   "+values.getValue());	

				//out$hessian <- optimHess(out$par, Qf) !!! Missing in Java
				//value.of.Q <- -out$objective
				valueOfQ = -values.getValue();			
				
				//shes <- try(solve(out$hessian))
				//se <- if (any(class(shes)%in%"try-error")) 
				//rep(NA_real_,2) else sqrt(diag(shes))
				} else {
					
					//System.out.println("k optimizacii");			  
					//org.apache.commons.lang3.time.StopWatch watch = new org.apache.commons.lang3.time.StopWatch();
					//watch.start();
					
					rConnection.assingVar("par", new double[]{logsig2e, logsig2b});
							rConnection.voidEval("out <- nlminb(start = par, objective = Qf, lower = c(-20, -20),  upper = c(20, 20))");
							logsig2e = rConnection.runEvalDoubles("out$par[1]")[0];
							logsig2b = rConnection.runEvalDoubles("out$par[2]")[0];
							
					//watch.stop();
					//System.out.println(watch.getTime()/100);
					//watch.reset();				
									
				}
				
			}
		}
		
        //fv <- solve(exp(-out$par[1])*W + exp(-out$par[2])*G,(exp(-out$par[1])*W)%*% as.vector(y))
//		solver = new QRDecomposition(wM.scalarMultiply(
//						FastMath.exp(-logsig2e)).add(gM.scalarMultiply(
//										FastMath.exp(-logsig2b)))).getSolver();

//		fv = (ArrayRealVector) solver.solve(
//						wM.scalarMultiply(FastMath.exp(-logsig2e)).operate(y));
		
		rConnection.assingVar("logsig2e", new double[]{logsig2e});
		rConnection.assingVar("logsig2b", new double[]{logsig2b});
		fv = new ArrayRealVector(rConnection.runEvalDoubles(
                "fv <- solve(exp(-logsig2e)*W + exp(-logsig2b)*G,(exp(-logsig2e)*W)%*% as.vector(y))"));
		
		
		
		sigmasHash.put(whichDistParameter, 
				new Double[]{FastMath.exp(logsig2e), FastMath.exp(logsig2b)});
		
        //b <- diff(fv, differences=order)
	    b = MatrixFunctions.diffV(fv, Controls.RW_ORDER);
	    
		//tr1 <- order + sum(b^2)/(exp(out$par[2])) # this always right 
	    //attributes(tr1) <- NULL
	    double tr1 = Controls.RW_ORDER + MatrixFunctions.sumV(
	    						b.ebeMultiply(b)) / (FastMath.exp(logsig2b));

	    tempV = null;
		return y.subtract(fv);
	}
	
	/**
	 * Inner class is a shell for objective function to
	 * find optimal lambda, for the case when neither. 
	 * logsig2e and logsig2b is fixed
	 *
	 */
	class NonLinObjFunction implements MultivariateFunction {
	    /** respnse variable .*/  
		private ArrayRealVector response;
		/** weights .*/
		private ArrayRealVector weights;
		/** object of ConnectionToR class. */
		private ConnectionToR rConnection;

		public double value(final double[] point) {
			return qF (point, response, weights, rConnection);
		}
		
		
		public void setResponse(ArrayRealVector response) {
			this.response = response;
		}
		
		public void setWeights(ArrayRealVector weights) {
			this.weights = weights;
		}
		
		public void setrConnection(ConnectionToR rConnection) {
			this.rConnection = rConnection;
		}
		
	}
	
	/**
	 * Inner class is a shell for objective function to
	 * find optimal lambda, for the case when 
	 * logsig2e is fixed.
	 *
	 */
	class NonLinObjFunction1 implements MultivariateFunction {
	    /** respnse variable .*/  
		private ArrayRealVector response;
		/** weights .*/
		private ArrayRealVector weights;
		/** */
		private double logsig2e;
		/** object of ConnectionToR class. */
		private ConnectionToR rConnection;
		
		public double value(final double[] point) {
			return qF1 (point, logsig2e, response, weights, rConnection);
		}
		
		public void setResponse(ArrayRealVector response) {
			this.response = response;
		}
		
		public void setWeights(ArrayRealVector weights) {
			this.weights = weights;
		}
		
		public void setLogsig2e(double logsig2e) {
			this.logsig2e = logsig2e;
		}
		
		public void setrConnection(ConnectionToR rConnection) {
			this.rConnection = rConnection;
		}
	}
	

	/**
	 * Inner class is a shell for objective function to
	 * find optimal lambda, for the case when 
	 * logsig2b is fixed.
	 *
	 */
	class NonLinObjFunction2 implements MultivariateFunction {
	    /** respnse variable .*/  
		private ArrayRealVector response;
		/** weights .*/
		private ArrayRealVector weights;
		/** */
		private double logsig2b;
		/** object of ConnectionToR class. */
		private ConnectionToR rConnection;
	
		public double value(final double[] point) {
			return qF2 (point, logsig2b, response, weights, rConnection);
		}
		
		public void setResponse(ArrayRealVector response) {
			this.response = response;
		}
		
		public void setWeights(ArrayRealVector weights) {
			this.weights = weights;
		}
		
		public void setLogsig2b(double logsig2b) {
			this.logsig2b = logsig2b;
		}
		
		public void setrConnection(ConnectionToR rConnection) {
			this.rConnection = rConnection;
		}
	}
	
	
	/**
	 * The objective function RW smoothing algorithm optimisation problem, 
	 * case when logsig2e is fixed.
	 */
	//Qf <- function(par)
	private double qF1 (double[] par, double logsig2e, ArrayRealVector y, 
								ArrayRealVector w, ConnectionToR rConnection) {
		
		return qF(new double[]{logsig2e, par[0]}, y, w, rConnection);
	}
	
	/**
	 * The objective function RW smoothing algorithm optimisation problem, 
	 * case when logsig2b is fixed.
	 */
	//Qf <- function(par)
	private double qF2 (double[] par, double logsig2b, ArrayRealVector y, 
								ArrayRealVector w, ConnectionToR rConnection) {
		
		return qF(new double[]{par[0], logsig2b}, y, w, rConnection);
	}
	
	/**
	 * The objective function RW smoothing algorithm optimisation problem.
	 */
	//Qf <- function(par)
	private double qF (double[] par, ArrayRealVector y, ArrayRealVector w, ConnectionToR rConnection) {
System.out.println("in");    
		//fv <- solve((exp(-par[1])*W) + (exp(-par[2])*G),(exp(-par[1])*W)%*%as.vector(y))		
		rConnection.assingVar("par", par);
		fv = new ArrayRealVector(rConnection.runEvalDoubles(
                "fv <- solve((exp(-par[1])*W) + (exp(-par[2])*G),(exp(-par[1])*W)%*%as.vector(y))"));		
		
		//b <- diff(fv, differences=order)
	    b = MatrixFunctions.diffV(fv, Controls.RW_ORDER);

	    //DT <-  determinant((exp(-par[1])*W) + (exp(-par[2])*G))$modulus
	    double dt = rConnection.runEvalDoubles(
	    		"DT <-  determinant((exp(-par[1])*W) + (exp(-par[2])*G))$modulus")[0];
	    
	    tempV = y.subtract(fv);
	    
		//f <- -(N/2)*log(2*pi*exp(par[1])) +           # 1 
	    double f = -(n / 2) * FastMath.log(2 * FastMath.PI 
	    * FastMath.exp(par[0]))
  
	    //               .5*sum(log(weights))-                   # 2 
	    + 0.5 * MatrixFunctions.sumV(MatrixFunctions.logVec(w))
	    
	    //               sum(weights*(y-fv)^2)/(2*exp(par[1])) + # 3 
	    - MatrixFunctions.sumV(w.ebeMultiply(tempV).ebeMultiply(tempV)) 
	    / (2 * FastMath.exp(par[0]))
	    
	    //               (N/2)*log(2*pi) -                       # 4
	    + (n * FastMath.log(2 * FastMath.PI)) / 2
	    
	    //               ((N-order)/2)*log(2*pi*exp(par[2]))-    # 5 
	    - ((n - Controls.RW_ORDER) * FastMath.log(2 * FastMath.PI 
	    * FastMath.exp(par[1]))) / 2
	    
	    //              sum(b^2)/(2*exp(par[2])) -              # 6 
	    - MatrixFunctions.sumV(b.ebeMultiply(b)) / (2 * FastMath.exp(par[1]))
	    
	    //               .5*DT								   # 7	
	    - 0.5 * dt;
	    
	    //attributes(f) <- NULL
	    //Pen <- if (penalty==TRUE) delta[1]*(par[1]-shift[1])^2 
	    //+ delta[2]*(par[2]-shift[2])^2 else 0 -f+Pen 
	    double pen = 0;
	    if (Controls.PENALTY) {
	    	 pen = delta.getEntry(0) * (par[0] - Controls.RW_SHIFT[0]) 
	    			 * (par[0] - Controls.RW_SHIFT[0]) + delta.getEntry(1)
	    			   					  * (par[1] - Controls.RW_SHIFT[1]) 
	    			   						 * (par[1] - Controls.RW_SHIFT[1]);
	    }
System.out.println("out");	   
	    //-f+Pen  # no penalty  yet    
	    return (-f + pen);
	}

	@Override
	public ArrayRealVector getVar() {
		// TODO Auto-generated method stub
		return null;
	}
}
