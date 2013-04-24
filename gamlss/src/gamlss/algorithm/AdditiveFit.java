/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;

import gamlss.utilities.Controls;
import gamlss.utilities.MatrixFunctions;
import gamlss.utilities.WLSMultipleLinearRegression;
import gamlss.utilities.oi.ConnectionToR;
import gamlss.distributions.BCPE;
import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GA;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.distributions.GT;
import gamlss.distributions.JSUo;
import gamlss.distributions.NO;
import gamlss.distributions.PE;
import gamlss.distributions.SST;
import gamlss.distributions.ST1;
import gamlss.distributions.ST3;
import gamlss.distributions.ST4;
import gamlss.distributions.TF;
import gamlss.distributions.TF2;
import gamlss.smoothing.GAMLSSSmoother;
import gamlss.smoothing.PB;
import gamlss.smoothing.RW;
import gamlss.smoothing.RWcallR;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.FastMath;

import java.util.HashMap;
import java.util.Hashtable;

/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public class AdditiveFit {

	/** z = residuals + fittedValues. */
	private ArrayRealVector z;
	/** vector of weights. */
	private ArrayRealVector w;
	/** residuals - the error in a result. */
	private ArrayRealVector residuals;
	/**	 Fitted values - fitted values of distribution parameter. */
	private ArrayRealVector fittedValues;
	/** Previously calculated values of a certain column of smoother matrix.  */
	private ArrayRealVector old;
	/** Number of iterations. */
	private int nit;
	/** currently fitting distribution parameter. */
	private int whichDistParameter;
	/** Variance of the smoothers. */
	private BlockRealMatrix var;
	/** AdditiveFit convergence parameter, used to determine a better model. */
	private double ratio;
	/** deltaf = deltaf + weighted.mean((s[, j] - old)^2, w). */
	private double deltaf;
	/** Penatly matrix. */
//	private Hashtable<Integer, BlockRealMatrix> penaltyM;
	/** degrees of freedom for each of distr parameter.. */
//	private HashMap<Integer, Integer> dfValues;
	/** base matrix. */
//	private Hashtable<Integer, BlockRealMatrix>  basisM;
	/** Smoother matrix. */
	private BlockRealMatrix s;
	/**	Object of Mean class. */
	private Mean mean;
	/** Temporary array for interim operations. */	
	private ArrayRealVector tempV;
	/** Temporary matrix for interim operations. */
	private BlockRealMatrix tempM;
	/** Temporary array for interim operations. */
	private double[] tempArr;
	/** Temporary Double for interim operations. */
	private Double tempD;
	/** Temporary int for interim operations. */
	private int size;
	/** i'th column of the smoother matrix. */
	private int colNum;
	/** Object of GAMLSSSmoother class. */
	private GAMLSSSmoother smoother;
	/** Object of ConnectionToR class. */
	private static ConnectionToR rConnection;
	/** Object of WLSMultipleLinearRegression class. */
	private WLSMultipleLinearRegression wls;
	//double rss;
	//double rssOld;

	/**
	 * Constructor of AdditiveFit class.
	 * @param distribution - object of the fitted 
	 * distribution of to the gamlss family
	 **@param  smoothMatrices - initially supplied smoother matrices
	 * each of the distribution parameterss
	 * @param smoothMatrices
	 * @param wlsReg - Object of WLSMultipleLinearRegression class. 
	 */
	public AdditiveFit(final GAMLSSFamilyDistribution distribution,
					   final HashMap<Integer, BlockRealMatrix> smoothMatrices,
					   final WLSMultipleLinearRegression wlsReg) {
		
		mean = new Mean();
		rConnection = new ConnectionToR();
		this.wls = wlsReg;
		
		smoother = null;
		switch (Controls.SMOOTHER) {
	       case Controls.PB:
	    	   smoother = new PB(distribution, rConnection, smoothMatrices);
	        break;
	       case Controls.RW:
	    	   if (Controls.BIG_DATA) {
		    	   smoother = new RWcallR(
		    			   distribution, rConnection, smoothMatrices);
	    	   } else {
	    		   smoother = new RW(distribution, smoothMatrices);
	    	   }
	    	   break;
	       default: 
				System.err.println("The specific smoother " 
						+ "has not been implemented yet in Gamlss!");
			}
		
	}

	/**
	 * Performs a smoothing process - creates an approximating function
	 * that attempts to capture important patterns in the data, while 
	 * leaving out noise or other fine-scale structures/rapid phenomena.
	 * @param xMat - design matrix
	 * @param y - response variable
	 * @param sMat - smoother matrix
	 * @param distParameter - distribution parameter
	 * @return matrix of smoothed values
	 */
	public BlockRealMatrix fitSmoother(final ArrayRealVector y, 
									   final ArrayRealVector weights,
									   final BlockRealMatrix xMat,
									   final BlockRealMatrix sMat, 
									   final int distParameter) {
		
		this.s = sMat.copy();
		this.w = weights;
		this.whichDistParameter = distParameter;
		
		//residuals <- as.vector(y - s %*% rep(1, ncol(s)))	
		tempV = new ArrayRealVector(s.getColumnDimension());
		tempV.set(1.0);
		tempV = (ArrayRealVector) s.operate(tempV);
		residuals = MatrixFunctions.vecSub(y, tempV);
		tempV 	= null;

		//fit <- list(fitted.values = 0)
		fittedValues = new ArrayRealVector(residuals.getDimension());
	
		//rss <- weighted.mean(residuals^2, w)
		//rss = calculateRss(residuals, w);
		//tempArr = null;
		
		//rssold <- rss * 10
		//rssOld = rss*10;
		
		//nit <- 0
		nit = 0;

		//df <- rep(NA, length(who))
        //lambda <- rep(NA, length(who))
		//double[] df = new double[s.getColumnDimension()]; 
		
		//var <- s
		var = s.copy();
		
        //if(trace) cat("\nADDITIVE   iter   rss/n     term\n")
		if (Controls.BACKFIT_TRACE) {
			System.out.println("ADDITIVE   iter   rss/n     term");
		}
		
		//ndig <- -log10(tol) + 1
		//double ndig = -FastMath.log10(tol)+1;
		
		//RATIO <- tol + 1
		ratio = Controls.BACKFIT_TOL + 1;
		
	    //while(RATIO > tol & nit < maxit) 
		while (ratio > Controls.BACKFIT_TOL 
										& nit < Controls.BACKFIT_CYCLES) {
	
			//rssold <- rss
			//rssOld = rss;
			
			//nit <- nit + 1
			nit++;
			
			//z <- residuals + fit$fitted.values			
			z = residuals.add(fittedValues);

//org.apache.commons.lang3.time.StopWatch watch = new org.apache.commons.lang3.time.StopWatch();
//watch.start();			
			
			//fit <- lm.wfit(x, z, w, method="qr", ...)
			wls.setNoIntercept(Controls.NO_INTERCEPT[whichDistParameter - 1]);
			wls.newSampleData(z, xMat.copy(), w.copy());
			
			wls.setNoIntercept(false);

			//residuals <- fit$residuals
			fittedValues = (ArrayRealVector) wls.calculateFittedValues(
															  Controls.IS_SVD);
//watch.stop();
//System.out.println(watch.getNanoTime()/(1e+9));
//watch.reset();			
	
			//residuals = z.subtract(fittedValues);
			//residuals = (ArrayRealVector) wls.calculateResiduals();
			residuals = z.subtract(fittedValues);
			
			//rss <- weighted.mean(residuals^2, w)
			//rss = calculateRss(residuals, w);
			
	        //if(trace) cat(nit, "   ", 
			//format(round(rss, ndig)), "  Parametric -- lm.wfit\n", sep = "")
			if (Controls.BACKFIT_TRACE) {
				//System.out.println(" " + nit + "  " + 
				//rss + " Parametric -- lm.wfit");
			}
			
			//deltaf <- 0
			deltaf = 0;
			
			//for(j in seq(names.calls)) 
			for (colNum = 0; colNum < s.getColumnDimension(); colNum++) {
			
				//old <- s[, j]
				old = (ArrayRealVector) s.getColumnVector(colNum);

				//z <- residuals + s[, j]
				z = residuals.add(old);
				
                //fit.call <- eval(smooth.calls[[j]])
				//residuals <- as.double(fit.call$residuals)		
				residuals = smoother.solve(this);
				
	            //if(length(residuals) != n)
				//stop(paste(names.calls[j], "returns a vector 
				//of the wrong length"))
				if (residuals.getDimension() != y.getDimension()) {
					System.err.println(colNum + "  column of matrix s has a"
							+ " vector of the wrong length");
				}

				//s[, j] <- z - residual
				s.setColumnVector(colNum, z.subtract(residuals));

				//if (length(fit.call$lambda)>1)
	            //{for cases where there are multiple lambdas 
	            //ambda[j] <- fit.call$lambda[1] 
				
				//coefSmo[[j]] <- if(is.null(fit.call$coefSmo))
				//0 else fit.call$coefSmo
				
				//deltaf <- deltaf + weighted.mean((s[, j] - old)^2, w)
				tempV = MatrixFunctions.vecSub((ArrayRealVector) 
											   s.getColumnVector(colNum), old);
				deltaf = deltaf + mean.evaluate(
							  tempV.ebeMultiply(tempV).getDataRef(), w.getDataRef());
				tempV = null;
		
				//rss <- weighted.mean(residuals^2, w)
				//rss = calculateRss(residuals, w);
				
				// if(trace)
				if (Controls.BACKFIT_TRACE) {
	                //cat(" ", nit, " ", format(round(rss, ndig)), 
					//"  Nonparametric -- ", names.calls[j], "\n", sep = "")
					//System.out.println("   " + nit +"   " + rss + "
					//Nonparametric " + "pb(column "+ i+")");
				}
				
				//df[j] <- fit.call$nl.df
				//df[i] = pb.getEdf();
				
				//if(se)
				if (Controls.IS_SE) {
					//var[, j] <- fit.call$var
					var.setColumnVector(colNum, smoother.getVar());
				}
			}
				
			//RATIO <- sqrt(deltaf/sum(w * apply(s, 1, sum)^2))	
			tempD = 0.0;
			tempM = new BlockRealMatrix(s.getRowDimension(), 1);
			for (int j = 0; j < s.getRowDimension(); j++) {   
				for (int k = 0; k < s.getColumnDimension(); k++) {
					tempD = tempD + s.getEntry(j, k);
				}
				tempM.setEntry(j, 0, tempD);
				tempD = 0.0;
			} 
			size = tempM.getRowDimension();
			for (int j = 0; j < size; j++) {   
				tempD = tempD + tempM.getEntry(j, 0) 
							* tempM.getEntry(j, 0) * w.getEntry(j);
			}
			ratio = FastMath.sqrt(deltaf / tempD);
			tempD = null;
			tempM = null;	
			
	        //if(trace)
			//cat("Relative change in functions:", 
			//format(round(RATIO, ndig)), "\n")
	        if (Controls.BACKFIT_TRACE) {
	        	System.out.println("Relative change in functions:  " + ratio);
	        } 
		}
		
	   //if(nit == maxit && maxit > 1)
	   //warning(paste("additive.fit convergence not 
	   //obtained in ", maxit, " iterations"))
	   if (ratio > Controls.BACKFIT_TOL) {
		    System.out.println("AdditiveFit convergence is not obtained in "
		    					+ Controls.BACKFIT_CYCLES + " iterations");
	   }     
		
	   //fit$fitted.values <- y - residuals
	   fittedValues = y.subtract(residuals);

	   //rl <- c(fit, list(smooth = s, nl.df = 
	   //sum(df), lambda=lambda, coefSmo=coefSmo))
	   return s;
	}

	/**
	 * Get smoother matrix processed in AdditivFit.
	 * @return smoother matrix
	 */
	public final BlockRealMatrix getS() {
			return s;
	}

	/**
	 * Calculates rss = weighted.mean(residuals^2, w).
	 * @param resid - residuals
	 * @param weights - vector of weights
	 * @return weighted.mean(residuals^2, w)
	 */
	private double calculateRss(final ArrayRealVector resid,
											final ArrayRealVector weights) {
		//rss <- weighted.mean(residuals^2, w)
		size = resid.getDimension();
		tempArr = new double[size];
		for (int i = 0; i < size; i++) {   
			tempArr[i] = resid.getEntry(i) * resid.getEntry(i);
		} 
		size = 0;
		return mean.evaluate(new ArrayRealVector(
							tempArr, false).getDataRef(), weights.toArray());
	}
	
	/**
	 * Get fitted values of distribution parameter processed in AdditiveFit.
	 * @return fitted values
	 */
	public final ArrayRealVector getFittedValues() {
		return fittedValues;
	}
	
	/**
	 * Get the class (smoother) that implements GAMLSSSmoother.
	 * @return Smoother class object
	 */
	public final GAMLSSSmoother getSmoother() {
		return smoother;
	}
	
	/**
	 * Get z = residuals + fittedValues.
	 * @return z
	 */
	public final ArrayRealVector getZ() {
		return z;
	}
	
	/**
	 * Get weights.
	 * @return vector of weights
	 */
	public final ArrayRealVector getW() {
		return w;
	}
	
	/**
	 * Get the fitting distribution parameter.
	 * @return distribution parameter number
	 */
	public final int getWhichDistParameter() {
		return whichDistParameter;
	}
	
	/**
	 * Get i'th column of the smoother matrix.
	 * @return i'th column number
	 */
	public final int getColNum() {
		return colNum;
	}
}
