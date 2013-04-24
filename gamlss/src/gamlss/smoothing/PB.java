/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.smoothing;

import java.util.HashMap;
import java.util.Hashtable;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.analysis.solvers.UnivariateSolver;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.direct.BOBYQAOptimizer;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import gamlss.utilities.MatrixFunctions;
import gamlss.algorithm.AdditiveFit;
import gamlss.algorithm.RSAlgorithm;
import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.utilities.ArithmeticSeries;
import gamlss.utilities.Controls;
import gamlss.utilities.oi.ConnectionToR;



/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public class PB implements GAMLSSSmoother {

	/** gM = lambda * t(penaltyM) %*% penaltyM. */
	private BlockRealMatrix gM;
	/** xwM = w * basisM. */
	private BlockRealMatrix xwM;
	/** transpose of xwM .*/
	private BlockRealMatrix xwMt;
	/** xwxM = t(xwM) %*% basisM. */
	private BlockRealMatrix xwxM;
	/** solution of the linear equation A × H = B for matrices A.*/
	private BlockRealMatrix hM;
	/** S = transpose(penaltyM)*penaltyM. */
	private BlockRealMatrix sM;
	/** matrix R of the QR decomposition.  */
	private BlockRealMatrix rM;
	/** inverse matrix R of the QR decomposition.  */
	private BlockRealMatrix rMinv;
	/** augmented basisM matrix. */
	private BlockRealMatrix xaug;
	/** wxM = isM * basisM. */
	private BlockRealMatrix wxM;
	/** augmented w vector. */
	private ArrayRealVector waug;
	/** the hat matrix. */
	private ArrayRealVector lev;
	/** isM = square root of weights vector. */
	private ArrayRealVector isM;
	/** gamma differences. */
	private ArrayRealVector gamma;
	/** the variance of the smoothers. */
	private ArrayRealVector var;
	/** regression coefficient. */
	private ArrayRealVector beta;
	/** fitted values of the distribution parameter. */
	private ArrayRealVector fv;
	/** residuals - difference between the sample 
	 * and the estimated function value. */
	private ArrayRealVector residuals;
	/** real parts of the eigenvalues of the original matrix. */
	private double[] uduV;
	/** real parts of the eigenvalues of the original matrix. */
	private BlockRealMatrix uduM;
	/** eighen degree of freedom. */
	private double edf;
	/** sig2 = sum(w * (y - fv) ^ 2) / (n - edf). */
	private double sig2;
	/** tau2 = sum(gamma ^ 2) / (edf-order). */
	private double tau2;
	/**edfTemp1 = edf-df. */
	private double edfTemp1;
	/** edfTemp2 = edf-df. */
	private double edfTemp2;
	/** previously calculated value of lambda. */
	private double lambdaOld;
	/** the no of observations. */
	private int n;
	/** Object of DecompositionSolver class.*/
	private DecompositionSolver solver;
	/** temp lambda value used in interim operations. */
	private double lambdaS;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Temporary matrix for interim operations. */
	private BlockRealMatrix tempM;
	/** Temporary array for interim operations. */
	private double[] tempArr;
	/** Temporary double for interim operations. */
	private double tempD;
	/** object of non-linear optimisation objective function.  */
	private NonLinObjFunction nonLinObj;
	/** object of uni-root objective function. */
	private UniRootObjFunction uniRootObj;
	/** Object of BOBYQAOptimizer class.*/
	private BOBYQAOptimizer optimizer;
	

	/** Hashtable to store lambdas. */
	private Hashtable<Integer, Double> lambdas 
								   = new Hashtable<Integer, Double>();
	/** Hashtable to store base matrices for each column of 
	 * smoother matrix of each distribution parameter. */
	private Hashtable<Integer, Object> baseMatricesSet
								  = new Hashtable<Integer, Object>();
	/** Hashtable to store penalty matrices for each column of 
	 * smoother matrix of each distribution parameter. */
	private Hashtable<Integer, Object> penaltyMatricesSet
	  							  = new Hashtable<Integer, Object>();
	/** Hashtable to store degree of freedom values for each column of 
	 * smoother matrix of each distribution parameter. */
	private HashMap<Integer, Object> dfValuesSet
								  = new HashMap<Integer, Object>();
	/** Hashtable to store base matrices for each column of 
	 * smoother matrix of single distribution parameter. */
	private Hashtable<Integer, BlockRealMatrix> baseMatrices 
								  = new Hashtable<Integer, BlockRealMatrix>();
	/** Hashtable to store penalty matrices for each column of 
	 * smoother matrix of single distribution parameter. */
	private Hashtable<Integer, BlockRealMatrix> penaltyMatrices 
								  = new Hashtable<Integer, BlockRealMatrix>();
	/** Hashtable to store degree of freedom values for each column of 
	 * smoother matrix of single distribution parameter. */
	private HashMap<Integer, Integer> dfValues 
								  = new HashMap<Integer, Integer>();
	/** Object of ConnectionToR class. */
	private static ConnectionToR rConnection;
	/** Fitting distribution parameter. */
	private int whichDistParameter;
	/** Hashtable of initially supplied smoother matrices. */
	private HashMap<Integer, BlockRealMatrix> smootherMatrices
							= new HashMap<Integer, BlockRealMatrix>();
//	/** Value of lambda. */
//	private Double lambda;
	/** column number of smoother matrix. */
	private int colNum;
	/** Vector of response variable values. */
	private ArrayRealVector y;
	/** Vector of weights. */
	private ArrayRealVector w;
	/** Basis matrix. */
	private BlockRealMatrix basisM;
	/** Penalty matrix. */
	private BlockRealMatrix penaltyM;
	/** Degrees of freedom. */
	private Integer df;
	
	
	
	/**
	 * Constructor of PB class.
	 * @param distr - object of the fitted distribution belonging
	 *  to the gamlss family
	 * @param rConnect - object of ConnectionToR class
	 * @param smoothMatrices - initially suplied smoother matrices for each 
	 * of the distribution parameters
	 */
	public PB(final GAMLSSFamilyDistribution distr, 
			  final ConnectionToR rConnect, 
			  final HashMap<Integer, BlockRealMatrix> smoothMatrices) {
		
		nonLinObj = new NonLinObjFunction();
		uniRootObj = new UniRootObjFunction();
		optimizer = new BOBYQAOptimizer(Controls.BOBYQA_INTERPOL_POINTS);
		this.rConnection = rConnect;
		this.smootherMatrices = smoothMatrices;
		
		for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
			if (smoothMatrices.get(i) != null) { 
//				if (smoothMatrices.containsKey(i)) {
					// Create smoother matrices of zeros
						lambdas.put(i, Controls.INITIAL_LAMBDA);
					for (int j = 0; j < smoothMatrices.get(
												i).getColumnDimension(); j++) {
						buildMatrices((ArrayRealVector) 
								smoothMatrices.get(i).getColumnVector(j), 
								Controls.DF_USER_DEFINED[i - 1], j);
					}
					baseMatricesSet.put(i, baseMatrices.clone());
					penaltyMatricesSet.put(i, penaltyMatrices.clone());
					dfValuesSet.put(i, dfValues.clone());
					baseMatrices.clear();
					penaltyMatrices.clear();
					dfValues.clear();
					
//				}
			}
		}
	}
    
	/**
	 * The main fitting method, initiate appropriate smoothing 
	 * function according to incoming parameters.
	 * @param additiveFit -object of AdditiveFoit class
	 * @return reiduals
	 */
  	//gamlss.pb <- function(x, y, w, xeval = NULL, ...)		
  		public ArrayRealVector solve(final AdditiveFit additiveFit) {
  			
  		Double lambda = Controls.LAMBDAS_USER_DEFINED;
  		colNum = additiveFit.getColNum();
  		y = additiveFit.getZ();
  		w = additiveFit.getW();
  		whichDistParameter =  additiveFit.getWhichDistParameter();
  		basisM = (BlockRealMatrix) getBasisM().get(colNum);
  		penaltyM = (BlockRealMatrix) getPenaltyM().get(colNum);
  		df = (Integer) getDfValues().get(colNum);
  		 		
  		//n <- nrow(X) # the no of observations
  		n = basisM.getRowDimension();
  		  		
  		//lambdaS <-  get(startLambdaName, envir=gamlss.env)
  		//geting the starting value
  		lambdaS = getLambdas().get(whichDistParameter);
  		
  		//if (lambdaS>=1e+07) lambda <- 1e+07
  		if (lambdaS >= 1e+07) {
  			 lambda = 1e+07;
  		}
  		//if (lambdaS<=1e-07) lambda <- 1e-07
  		if (lambdaS <= 1e-07) {
  			 lambda = 1e-07;
  		}
  		   		  		
  		//if (is.null(df)&&!is.null(lambda)||!is.null(df)&&!is.null(lambda))
	  	if (lambda != null) {
	  		
	  			//fit <- regpen(y, X, w, lambda, D)
	  			beta = regpen(lambda);
	  			
	  			//fv <- X %*% fit$beta
	  			fv = (ArrayRealVector) basisM.operate(beta);
	  			
	  	//else if (is.null(df)&&is.null(lambda))
	  	} else if (df == null) {
	  		
	  			//lambda <- lambdaS
	  			lambda = lambdaS;
	
	  			//switch(control$c," ML"={
	  			switch (Controls.SMOOTH_METHOD) {
	  		       case DistributionSettings.GAIC:
	  		    	 	fv = functionGAIC(lambda);
	  			     break;
	  		       case DistributionSettings.ML:	 
	  		           fv = functionML(lambda);
	  		         break;
	  		       case DistributionSettings.ML1:
	  		    	   fv = functionML1(lambda);
	  		    	 break;
	  		       case DistributionSettings.EM:
	  		    	   	System.err.println("EM has not been implemented yet");
	  		    	 break;
	  		       case DistributionSettings.GCV:
	  		    	   fv = functionGCV(lambda);
	  			     break;
	  		     default: 
						System.err.println(" Cannot recognise the " 
										+ "smoothing method or it has"
						        			+ "not been implemented yet");
	  			}
	  		} else {	
	  			
	  		//QR <- qr(sqrt(w)*X)
	  		//Rinv <- solve(qr.R(QR))
	  		rM = (BlockRealMatrix) new QRDecomposition(
	  					MatrixFunctions.multVectorMatrix(
	  							   MatrixFunctions.sqrtVec(w), basisM)).getR();
	  		rM = rM.getSubMatrix(0, rM.getColumnDimension() - 1,
	  										   0, rM.getColumnDimension() - 1);
	  		rMinv = (BlockRealMatrix) 
	  						  new QRDecomposition(rM).getSolver().getInverse();

	  		//S   <- t(D)%*%D
	  		sM = penaltyM.transpose().multiply(penaltyM);
	  		
	  		//UDU <- eigen(t(Rinv)%*%S%*%Rinv)
	  		uduV =  new EigenDecomposition(
	  				       rMinv.transpose().multiply(sM).multiply(rMinv),
	  				        	Controls.SPLIT_TOLERANCE).getRealEigenvalues();
  		
	  		//lambda <- if (sign(edf1_df(0))==sign(edf1_df(100000))) 100000
	  		//in case they have the some sign
  			edfTemp1 = edf1_df(0, df);
  			edfTemp2 = edf1_df(100000.0, df);
  			
  			if (FastMath.signum(edfTemp1) == FastMath.signum(edfTemp2)) {
  				  lambda = 100000.0;
  			} else {
  				
  				//else uniroot(edf1_df, c(0,100000))$root
  				uniRootObj.setDf(df);
  				final double relativeAccuracy = 1.0e-12;
  				final double absoluteAccuracy = 1.0e-8;
  				UnivariateSolver uniRootSolver 
  						 = new BrentSolver(relativeAccuracy, absoluteAccuracy);
  				lambda = uniRootSolver.solve(1000, uniRootObj, 0.0, 100000.0);
  			}
  		
  			//fit <- regpen(y, X, w, lambda, D)
  			beta = regpen(lambda);
  			fv = (ArrayRealVector) basisM.operate(beta);			
  		}	
  		if (Controls.IF_NEED_THIS) {
	  	
	  		//waug <- as.vector(c(w, rep(1,nrow(D))))
		  	waug = w.append(MatrixFunctions.repV(1.0, 
		  										  penaltyM.getRowDimension()));
		  	
	  			  
	  		//xaug <- as.matrix(rbind(X,sqrt(lambda)*D))
		  	xaug = MatrixFunctions.appendMatricesRows(basisM, 
		  		 (BlockRealMatrix) penaltyM.scalarMultiply(
		  				 							   FastMath.sqrt(lambda)));
		  	
		  	//lev <- hat(sqrt(waug)*xaug,intercept=FALSE)[1:n]
		  	//get the hat matrix
		  	lev = (ArrayRealVector) MatrixFunctions.getMainDiagonal(
		  			    new BlockRealMatrix(MatrixFunctions.calculateHat(
		  			    				 MatrixFunctions.multVectorMatrix(
		  			    					   MatrixFunctions.sqrtVec(waug), 
		  			    				 xaug)).getData())).getSubVector(0, n);
		  	
		  	//lev <- (lev-.hat.WX(w,x))
		  	//subtract  the linear since is already fitted 
		  	lev = lev.subtract(hatWX((ArrayRealVector) 
		  						getSmootherMatrices().get(
		  				whichDistParameter).getColumnVector(colNum)));
		  	
			// var <- lev/w
		  	//the variance of the smootherz
			var = lev.ebeDivide(w);
  		}
 			  
  			  
  		//if (is.null(xeval)) # if no prediction
  		if (Controls.XEVAL_USER_DEFINED == null) {
  			// Residuals
  			return y.subtract(fv);
  			
  		//else # for prediction
  		} else {
  			
  			//ll <- dim(as.matrix(attr(x,"X")))[1]  			      
  			//nx <- as.matrix(attr(x,"X"))[seq(length(y)+1,ll),]
  			tempArr = ArithmeticSeries.getSeries(y.getDimension() + 1, 
  												  basisM.getRowDimension(), 1);
  			BlockRealMatrix nx = new BlockRealMatrix(tempArr.length, 
  												  basisM.getColumnDimension());
  			for (int i = 0; i < tempArr.length; i++) {
  				nx.setRowVector(i, basisM.getRowVector((int) tempArr[i]));
  			}
  				  
  			//pred <- drop(nx %*% fit$beta)
  			ArrayRealVector pred = (ArrayRealVector) nx.operate(beta);
  		
  			// Residuals
  			return y.subtract(fv);
  		}
  	}
  	
  	/**
  	 *<p>Compute the "hat" matrix.
  	 * </p>
  	 * <p>The hat matrix is defined in terms of the design matrix X
  	 *  by X(X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>
  	 * </p>
  	 * @param xDesign - column of initial smoother matrix
  	 * @return hat matrix
  	 */
  	//function (w, x) 
  	private ArrayRealVector hatWX(final ArrayRealVector xDesign) {
  	    //p <- length(x)
  		int p = xDesign.getDimension();
  	    //X <- if (!is.matrix(x)) 
  		// matrix(cbind(1, x), ncol = 2)
  		//else x
  		tempM = new BlockRealMatrix(p, 2);
  		tempM.setColumnVector(0, MatrixFunctions.repV(1, p));
  		tempM.setColumnVector(1, xDesign);
  		
  		//k <- length(w)
  		//p <- dim(X)[1]
  		//if (p != k)
  		if (p != w.getDimension()) {
  			System.err.println("`w' and 'x' do not have the same length");
  			//stop("`w' and 'x' are not having the same length")
  		}
  		
  		//Is <- sqrt(w)
  	    isM = MatrixFunctions.sqrtVec(w);
  		
  	    //if (any(!is.finite(Is))) 
  	    if (isM.isInfinite()) {
  	    	//warning("diagonal weights has non-finite entries")
  	    	System.err.println("diagonal weights has non-finite entries");
  	    }
  	    
  	    //WX <- X
  	    //wxM = tempM;

  		//WX[] <- Is * X
  	    wxM = MatrixFunctions.multVectorMatrix(isM, tempM);
  	    
  	    //h<-hat(qr(WX))
  	    return MatrixFunctions.getMainDiagonal((BlockRealMatrix) new 
  	    		BlockRealMatrix(MatrixFunctions.calculateHat(wxM).getData()));
  	}
  	
  	/**
  	 * local function to get df using eigen values.
  	 * @param lambda - smoothing parameter
  	 * @param df - degree of freedom
  	 * @return sum(1/(1+lambda*UDU$values) - df
  	 */
  	//edf1_df <- function(lambda)
  	private double edf1_df(final double lambda, final double df) {
  	//edf <- sum(1/(1+lambda*UDU$values))
  		  double out = 0;
  		  for (int i = 0; i < uduV.length; i++) {
  		    out = out + (1 / (1 + lambda * uduV[i]));
  		  }	  
  		  //(edf-df)
  		  return  out - df;
  	}
  	
  	/**
  	 * ML smoothing method.
  	 * @param lambda - smoothing parameter
  	 * @return fitted values
  	 */
  	private ArrayRealVector functionML(Double lambda) { 
  		
  		//for (it in 1:50)
  		for (int i = 0; i < Controls.ML_ITER; i++) {
  			
  			//fit <- regpen(y, X, w, lambda, D)
  			//fit model
  			beta = regpen(lambda);
  			
  			//gamma. <- D %*% as.vector(fit$beta)
  			//get the gamma differences
  			gamma = (ArrayRealVector) penaltyM.operate(beta);
 
  			//fv <- X %*% fit$beta
  			//fitted values
  			fv = (ArrayRealVector) basisM.operate(beta);  			 

  			//sig2 <- sum(w * (y - fv) ^ 2) / (n - fit$edf)
  			tempV = MatrixFunctions.vecSub(y, fv);
  			sig2 = MatrixFunctions.sumV(
  					            w.ebeMultiply(tempV.ebeMultiply(tempV)))
  					   								    	  / (n - getEdf());
  			tempV = null;
  			
  			//tau2 <- sum(gamma. ^ 2) / (fit$edf-order)
  			tau2 = MatrixFunctions.sumV(gamma.ebeMultiply(gamma))
  									            / (getEdf() - Controls.ORDER);
  			
  			//if(tau2<1e-7) tau2 <- 1.0e-7
  			if (tau2 < 1e-7) {
  				tau2 = 1e-7;
  			}
  			
  			//lambda.old <- lambda
  			lambdaOld = lambda;
		
  			//lambda <- sig2 / tau2
  			lambda = sig2 / tau2; 	

  			//if (lambda<1.0e-7) lambda<-1.0e-7 
  			if (lambda < 1.0e-7) {
  				lambda = 1.0e-7;
  			}
  			//if (lambda>1.0e+7) lambda<-1.0e+7
  			if (lambda > 1.0e7) {
  				lambda = 1.0e7;
  			}
  			
  			//if (abs(lambda-lambda.old) < 1.0e-7        ||lambda>1.0e10) break
  			if (FastMath.abs(lambda - lambdaOld) < 1.0e-7 || lambda > 1.0e10 ) {
  				break;
  			}
  		}	
		setLambda(lambda);	
		return fv;
  	}
  	
  	/**
  	 * ML1 smoothing method.
  	 * @param lambda - smoothing parameter
  	 * @return fitted values
  	 */
  	private ArrayRealVector functionML1(Double lambda) {
  		
  		
  		//for (it in 1:50)
  		for (int i = 0; i < Controls.ML1_ITER; i++) {
  			
  			//fit <- regpen(y, X, w, lambda, D)
  			//fit model
  			beta = regpen(lambda);
  			
  			//gamma. <- D %*% as.vector(fit$beta)
  			//get the gamma differences
  			gamma = (ArrayRealVector) penaltyM.operate(beta);
 
  			//fv <- X %*% fit$beta
  			//fitted values
  			fv = (ArrayRealVector) basisM.operate(beta);  			 

  			//sig2 <- 1
  			sig2 = 1.0;
  			
  			//tau2 <- sum(gamma. ^ 2) / (fit$edf-order)
  			tau2 = MatrixFunctions.sumV(gamma.ebeMultiply(gamma))
  									            / (getEdf() - Controls.ORDER);
  			
  			//if(tau2<1e-7) tau2 <- 1.0e-7
  			if (tau2 < 1e-7) {
  				tau2 = 1e-7;
  			}
  			
  			//lambda.old <- lambda
  			lambdaOld = lambda;
		
  			//lambda <- sig2 / tau2
  			lambda = sig2 / tau2; 	

  			//if (lambda<1.0e-7) lambda<-1.0e-7 
  			if (lambda < 1.0e-7) {
  				lambda = 1.0e-7;
  			}
  			//if (lambda>1.0e+7) lambda<-1.0e+7
  			if (lambda > 1.0e7) {
  				lambda = 1.0e7;
  			}
  			
  			//if (abs(lambda-lambda.old) < 1.0e-7        ||lambda>1.0e7) break
  			if (FastMath.abs(lambda - lambdaOld) < 1.0e-7 || lambda > 1.0e7 ) {
  				break;
  			}
  		}	
		setLambda(lambda);	
		return fv;
  	}


	/**
	 * Simple penalised regression.
	 * @return betas - estimates resulting from an analysis performed
	 */
  	//regpen <- function(y, X, w, lambda, D)
  	private ArrayRealVector regpen(Double lambda) {
   
  		//G <- lambda * t(D) %*% D
  		gM = (BlockRealMatrix) penaltyM.transpose().multiply(
  											  penaltyM).scalarMultiply(lambda);
  		//XW <- w * X
  		xwM = MatrixFunctions.multVectorMatrix(w, basisM);
  		
  		//XWX <- t(XW) %*% X
  		xwMt = xwM.transpose();
  		xwxM = xwMt.multiply(basisM);
  		
  		//beta <- solve(XWX + G, t(XW) %*% y)
  		//solver = new SingularValueDecomposition(xwxM.add(gM)).getSolver();
  		solver = new QRDecomposition(xwxM.add(gM)).getSolver();
  		beta = (ArrayRealVector) 
  						solver.solve((ArrayRealVector) xwMt.operate(y));
  		
  		//H <- solve(XWX + G, XWX)
  		hM = new BlockRealMatrix(solver.solve(xwxM).getData());

        //edf <- sum(diag(H))
        //fit <- list(beta = beta, edf = sum(diag(H)))
		setEdf(hM.getTrace());
		
  		return beta;
  	}
  	
 	/**
  	 * GCV smoothing method.
  	 * @param lambda - smoothing parameter
  	 * @return fitted values
  	 */
		private ArrayRealVector functionGCV(Double lambda) {
			
			//QR <-qr(sqrt(w)*X)
	  		//Rinv <- solve(qr.R(QR))
			QRDecomposition qr = new QRDecomposition(
								MatrixFunctions.multVectorMatrix(
										  MatrixFunctions.sqrtVec(w), basisM));
	  		rM = (BlockRealMatrix) qr.getR();
	  		rM = rM.getSubMatrix(0, rM.getColumnDimension() - 1, 
	  										 0, rM.getColumnDimension() - 1);
	  		rMinv = (BlockRealMatrix) 
	  						  new QRDecomposition(rM).getSolver().getInverse();
			
	        //wy <- sqrt(w)*y
	  		ArrayRealVector wy = MatrixFunctions.sqrtVec(w).ebeMultiply(y); 
	  		
	  		//y.y <- sum(wy^2)
	  		double y_y = MatrixFunctions.sumV(wy.ebeMultiply(wy));
	  		
	  		//S   <- t(D)%*%D
	  		sM = penaltyM.transpose().multiply(penaltyM);
	  		
	  		//UDU <- eigen(t(Rinv)%*%S%*%Rinv)
	  		uduM =  new BlockRealMatrix(new EigenDecomposition(
	  					rMinv.transpose().multiply(sM).multiply(rMinv), 
	  							   Controls.SPLIT_TOLERANCE).getV().getData());

	  		
	  		//yy <- t(UDU$vectors)%*%t(qr.Q(QR))%*%wy
	  		BlockRealMatrix qM = (BlockRealMatrix) qr.getQ();
	  		//SingularValueDecomposition svd = new SingularValueDecomposition(
	  		//MatrixFunctions.multVectorMatrix(MatrixFunctions.sqrtVec(w), basisM));
	  	    //BlockRealMatrix qM = new BlockRealMatrix(svd.getV().getData());
	  		
	  		//.... to finish !!!!!!!
	  		MatrixFunctions.matrixPrint(qM);	  		
		return null;
		}
		 	
  	/**
  	 * GAIC smoothing method.
  	 * @param lambda - smoothing parameter
  	 * @return fitted values
  	 */
		private ArrayRealVector functionGAIC(Double lambda) {

			//lambda <- nlminb(lambda, fnGAIC,  
			//lower = 1.0e-7, upper = 1.0e7, k=control$k)$par
			lambda = optimizer.optimize(Controls.BOBYQA_MAX_EVAL,
										nonLinObj, 
										GoalType.MINIMIZE, 
										new double[] {lambda, lambda},
										new double[] {Double.MIN_VALUE,
										Double.MIN_VALUE}, 
										new double[] {Double.MAX_VALUE,
										Double.MAX_VALUE}).getPoint()[0];
			
			if (lambda > 1.0e+7) {
				lambda = 1.0e+7;
			}
			if (lambda < 1.0e-7) {
				lambda = 1.0e-7;
			}
			
			//fit <- regpen(y=y, X=X, w=w, lambda=lambda, D)
			beta = regpen(lambda);
			
			//fv <- X %*% fit$beta     
			fv = (ArrayRealVector) basisM.operate(beta);
			
			//assign(startLambdaName, lambda, envir=gamlss.env)
			setLambda(lambda);
			return fv;
		}
		
		/**
		 * Inner class is a shell for objective function to
		 * find optimal lambda.
		 *
		 */
		class NonLinObjFunction implements MultivariateFunction {
			/**
			 * Implemenrted function.
			 * @param point - point value
			 * @return objective function values
			 */
			public double value(final double[] point) {
				return fnGAIC(point[0]);
			}
		}
		
	 	/**
	  	 * Inner class is a shell for objective function to
	  	 * find the root of the function.
	  	 *
	  	 */
		class UniRootObjFunction implements UnivariateFunction {
			
		    /** degree of freedom. */  
			private double df;
					
			/**
			 * This function is used to evaluate the objective function.
			 * @param x - income value to determine zero of the function
			 * @return value of the function
			 */
			public double value(final double x) {
				return edf1_df(x, df);
			}	
			
			/**
			 * Set degree of freedom.
			 * @param df - degree of freedom
			 */
			public void setDf(final double df) {
				this.df = df;
			}
			
		}
		
		/**
		 * The objective function GAIC smoothing method optimisation problem.
		 * @param lambda - smoothing parameter
		 * @return sum(w*(y-fv)^2)+k*fit$edf
		 */
		private double fnGAIC(Double lambda) {
			//fnGAIC <- function(lambda, k)
			
			//fit <- regpen(y=y, X=X, w=w, lambda=lambda, D)
			beta = regpen(lambda);
			
			//fv <- X %*% fit$beta
			fv = (ArrayRealVector) basisM.operate(beta);
			
			//GAIC <- sum(w*(y-fv)^2)+k*fit$edf
  			tempV = MatrixFunctions.vecSub(y, fv);
			return MatrixFunctions.sumV(
			                 w.ebeMultiply(tempV.ebeMultiply(tempV))) 
			            							   + Controls.K * getEdf();
		}
		
		/**
		 * Constructs the base and penalty matrices and also sets a
		 *  value for degree of freedom.
		 * @param colValues - values of certain column (colNumber) of
		 *  the smooth matrix of the fitting distribution parameter 
		 * @param df - degree of freedom
		 * @param colNumber - number of the supplied column (colValues)
		 *  of the fitting distribution parameter
		 */
		//pb<-function(x, df = NULL, lambda = NULL, control=pb.control(...), ...)
	    private void buildMatrices(final ArrayRealVector colValues, 
	    								    Integer df, final int colNumber) {	    	
		    //X <- bbase(x, xmin, xmax, control$inter, control$degree,
		    //control$quantiles) # create the basis
		    BlockRealMatrix xM = formX(colValues);
		    	
		    //D <- if(control$order==0) diag(r) else diff(diag(r),
		    //diff=control$order) # the penalty matrix
		    BlockRealMatrix dM = formD(xM);
		
		    //if(!is.null(df)) # degrees of freedom
		    if (df != null) {
		         //if (df>(dim(X)[2]-2))
		    	if (df > xM.getColumnDimension() - 2) {
		            //df <- 3;
		        	df = 3;
		        		
		        	//warning("The df's exceed the 
		        	//number of columns of the design
		        	//matrix", "\n",  "   they are set to 3") 
		            System.err.println("The df's exceed the number of columns "
		            		     + " of the design matrix, they are set to 3");
		    	}
		    	if (df < 0) {
		    		//if (df < 0)  warning("the extra df's are set to 0") 
		    		//df <- if (df < 0)  2  else  df+2
		            System.err.println("the extra df's are set to 2");
		            df = 2;
		        } else {
		        	df = df + 2;
		        } 
		    }
			baseMatrices.put(colNumber, xM);
			penaltyMatrices.put(colNumber, dM);
			dfValues.put(colNumber, df);
	    }
		    

	    /**
	     * Constructs the base matrix.
	     * @param colValues - values of the certain column of
		 *  the smooth matrix which corresponds to the 
		 *  currently fitting distribution parameter
	     * @return - base matrix
	     */
	    //bbase <- function(x, xl, xr, ndx, deg, quantiles=FALSE)
		private static BlockRealMatrix formX(final ArrayRealVector colValues) {
			
		    //control$inter <- if (lx<99) 10 else control$inter # 
		    //this is to prevent singularities when length(x) is small
		    if (colValues.getDimension() < 99) {
		    	Controls.INTER = 10; 
		    }
		    
		    //xl <- min(x)
		    double xl = colValues.getMinValue();
		    	
		    //xr <- max(x)
		    double xr = colValues.getMaxValue();
		    	
		    //xmax <- xr + 0.01 * (xr - xl)
		    double xmax = xr + 0.01 * (xr - xl);
		    	
		    //xmin <- xl - 0.01 * (xr - xl)
		    double xmin = xl - 0.01 * (xr - xl);
			
			//dx <- (xr - xl) / ndx
			double dx = (xmax - xmin) / Controls.INTER;
			
		
			//if (quantiles) # if true use splineDesign
			if (Controls.QUANTILES) { 
				//knots <-  sort(c(seq(xl-deg*dx, xl, dx),quantile(x, 
				//prob=seq(0, 1, length=ndx)), seq(xr, xr+deg*dx, dx))) 
				ArrayRealVector kts = null;
				
				//B <- splineDesign(knots, x = x, outer.ok = TRUE, ord=deg+1)
				//return(B)  
				return null;	
			} else {
			
				//kts <-   seq(xl - deg * dx, xr + deg * dx, by = dx)
				//ArrayRealVector kts = new ArrayRealVector(
				//ArithmeticSeries.getSeries(xl-deg*dx, xr+deg*dx, dx),false);
				
				rConnection.assingVar("min", 
									new double[]{xmin - Controls.DEGREE * dx});
				rConnection.assingVar("max", 
									new double[]{xmax + Controls.DEGREE * dx});
				rConnection.assingVar("step", new double[]{dx});
			
				ArrayRealVector kts 
						= new ArrayRealVector(rConnection.runEvalDoubles(
			    		                 "knots <- seq(min, max, by = step)"));
				
				//P <- outer(x, kts, FUN = tpower, deg)
				BlockRealMatrix pM = MatrixFunctions.outertpowerPB(colValues,
														kts, Controls.DEGREE);
				
				//D <- diff(diag(dim(P)[2]), 
				//diff = deg + 1) / (gamma(deg + 1) * dx ^ deg)
				BlockRealMatrix tempM = MatrixFunctions.diff(
							MatrixFunctions.buildIdentityMatrix(
								pM.getColumnDimension()), Controls.DEGREE + 1);
				
				double[][] tempArrArr = new double[tempM.getRowDimension()]
												  [tempM.getColumnDimension()];
				for (int i = 0; i < tempArrArr.length; i++) {
					 for (int j = 0; j < tempArrArr[i].length; j++) {
						 tempArrArr[i][j] = tempM.getEntry(i, j)
						 / ((FastMath.exp(Gamma.logGamma(Controls.DEGREE + 1)))
									       * FastMath.pow(dx, Controls.DEGREE));
					 }
				}
				tempM = new BlockRealMatrix(tempArrArr);	
		
				//B <- (-1) ^ (deg + 1) * P %*% t(D)
				return  (BlockRealMatrix) pM.multiply(
						   tempM.transpose()).scalarMultiply(FastMath.pow(
								                   -1, (Controls.DEGREE + 1)));
			}
	}
		
		/**
		 * Constructs the penalty matrix.
		 * @param xM - base matrix
		 * @return penalty matrix
		 */
	  	private BlockRealMatrix formD(final BlockRealMatrix xM) {
	      	tempArr = new double[xM.getColumnDimension()];
	          for (int i = 0; i < tempArr.length; i++) {
	          		tempArr[i] = 1.0;
	          }
	          if (Controls.ORDER == 0) {
	        	  return new BlockRealMatrix(
	        		  MatrixUtils.createRealDiagonalMatrix(tempArr).getData());
	          } else {
	        	  return MatrixFunctions.diff(new BlockRealMatrix(
	        			                MatrixUtils.createRealDiagonalMatrix(
	        			                  tempArr).getData()), Controls.ORDER);
	          }
	      }
	  	
		/**
		 * Get a set (for all distr parameters) of lambdas.
		 * @return set of lambdas
		 */
		public final Hashtable<Integer, Double> getLambdas() {
			return lambdas;
		}

		/**
		 * 
		 * @param lambda - single smoothing parameterer 
		 * attributes to the currently fitting distribution
		 *  parameter.
		 */
		public final void setLambda(final double lambda) {
			lambdas.put(whichDistParameter, lambda);
		}
		
		/**
		 * Get degree of freedom values for each column of smoother 
		 * matrix of each distribution parameter.
		 * @return - set of degree of freedom values
		 */
		public final HashMap<Integer, Object> getDfValuesSet() {
			return dfValuesSet;
		}

		/**
		 * Get a penalty matrix for each column of smoother 
		 * matrix of each distribution parameter.
		 * @return penalty matrix
		 */
		public final Hashtable<Integer, Object> getPenaltyMatricesSet() {
			return penaltyMatricesSet;
		}

		/**
		 * Get a base matrix for each column of smoother 
		 * matrix of each distribution parameter.
		 * @return base matrix
		 */
		public final Hashtable<Integer, Object> getBaseMatricesSet() {
			return baseMatricesSet;
		}
		
		/**
		 * Get a set of initially suppplied smoother matrices.
		 * @return set of initially suppplied smoother matrices
		 */
		public final HashMap<Integer, BlockRealMatrix> getSmootherMatrices() {
			return smootherMatrices;
		}
		
		/**
		 * Get basis matrix.
		 * @return basis matrix
		 */
		public final Hashtable getBasisM() {
			return (Hashtable) baseMatricesSet.get(whichDistParameter);
		}
		
		/**
		 * Get penalty matrix.
		 * @return penalty matrix
		 */
		public final Hashtable getPenaltyM() {
			return (Hashtable) penaltyMatricesSet.get(whichDistParameter);
		}
		
		/**
		 * Get degree of freedom.
		 * @return degree of freedom
		 */
		public final HashMap getDfValues() {
			return (HashMap) dfValuesSet.get(whichDistParameter);
		}
  	 
		/**
		 * Set eighen degree of freedom.
		 * @param edf - eighen degree of freedom
		 */
		public final void setEdf(final double edf) {
		this.edf = edf;
		}
		
		/**
		 * Get eighen degree of freedom.
		 * @return edf
		 */
		public final double getEdf() {
			return edf;
		}
	
		//public double getEdf() {
		//	return edf-2;
		//}
		
		/**
		 * Get variance.
		 * @return variance
		 */
		public final ArrayRealVector getVar() {
			return var;
		}
}