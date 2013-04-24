/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;

import java.util.HashMap;
import java.util.Hashtable;

import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.MatrixFunctions;
import gamlss.utilities.WLSMultipleLinearRegression;
import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.utilities.Controls;


import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;

/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */

public class GlimFit {

	/** Number of GlimFit iterations performed.  */
	private int itn;
	/** Boolean used to print the message when the deviance has increased. */
	private boolean iterw;
	/** Global deviance. */
	private double dv;
	/** Previously stored value of the global deviance. */
	private double olddv;
	/** Vector of linear predictor values. */
	private ArrayRealVector eta;
	/** Vector of linear predictor values. */
	private ArrayRealVector lp;
	/**  Previously stored vector of linear predictor values. */
	private ArrayRealVector lpold;
	/** Vector of 1/(link function of the linear predictor) values. */
	private ArrayRealVector dr;
	/** Vector of first derivative  values with respect to the
	 *  "whatDistParameter" parameter. */
	private ArrayRealVector dldp;
	/** Vector of second derivative  values with respect to the
	 *  "whatDistParameter" parameter. */
	private ArrayRealVector d2ldp2;
	/** wt = vector of values -(d2ldp2/(dr*dr)). */
	private ArrayRealVector wt;
	/** wv = vector of values (eta-os)+dldp/(dr*wt). */
	private ArrayRealVector wv;
	/** Values returned by WLSMultipleLinearRegression after linear fitting. */
	private ArrayRealVector fvLinear;
	/** Previously stored smoother matrix.*/
	private BlockRealMatrix sMatrixOld;
	/** Object of MakeLinkFunction class. */
	private MakeLinkFunction makelink;
	/** Object of WLSMultipleLinearRegression class. */
	private WLSMultipleLinearRegression wls;
	/** Object of AdditiveFit class. */
	private AdditiveFit additive;
	/** Temporary array for interim operations.*/
	private double[] tempArr;
	/** Hashtable to store smoother matrices. */
	private HashMap<Integer, BlockRealMatrix> sMatrices 
	   				= new HashMap<Integer, BlockRealMatrix>();
	/** Smother matrix. */
	private BlockRealMatrix sMatrix;
	 /** Vector of the fitted values of distribution parader.  */
	private ArrayRealVector fv;
	/** Object of the fitted distribution of to the gamlss family. */
	private GAMLSSFamilyDistribution distr;
	/** Vector of response variable values. */
	private ArrayRealVector response;
	/** Design matrix. */
	private BlockRealMatrix xMatrix;
	/** Design matrices for each of the distribution parameters. */
	private Hashtable<Integer, BlockRealMatrix> designMatrices;
	/** Vector of weights. */
	private ArrayRealVector w;
	/** Steps for the evaluation of the response variable. */
	private double step;
	/** offset value for the fitted distribution. */
	private double offSet;
	
	/**
	 * This is to emulate the GlimFit iterative algorithm.
	 * @param distribution - object of the fitted 
	 * distribution of to the gamlss family
	 * @param y -  vector of response variable values
	 * @param designM - design matrices for
	 **@param  smoothM - initially supplied smoother matrices
	 * each of the distribution parameters
	 * @param weights - vector of the weight values
	 */
	public GlimFit(final GAMLSSFamilyDistribution distribution,
			       final ArrayRealVector y, 
			       final Hashtable<Integer, BlockRealMatrix> designM,
			       final HashMap<Integer, BlockRealMatrix> smoothM,
			       final ArrayRealVector weights) {
		
		this.distr = distribution;
		this.response = y;
		this.designMatrices = designM;
		this.w = weights;
		
		makelink = new MakeLinkFunction();
		wls = new WLSMultipleLinearRegression(Controls.COPY_ORIGINAL);
		if (Controls.SMOOTHING) {
			additive = new AdditiveFit(distribution, smoothM, wls);
		}
	}

	/**
	 * Gamlss fitting algorithm.
	 *  @param whichDistParameter distribution parader
	 */
	public void glimFitFunctionRS(final int whichDistParameter) {
		
		sMatrix = sMatrices.get(whichDistParameter);
		fv = distr.getDistributionParameter(whichDistParameter);
		xMatrix = designMatrices.get(whichDistParameter);
		step = Controls.STEP[whichDistParameter - 1];
		offSet = Controls.OFFSET[whichDistParameter - 1];
		
		//itn <- 0
		itn = 0;
		
		//lp <- eta <- f$linkfun(fv)
		eta = makelink.link(distr.getDistributionParameterLink(
													  whichDistParameter), fv);
		lp  = eta.copy();

		//dr <- f$dr(eta), //dr <- 1/dr
		dr  = MatrixFunctions.inverse(makelink.distParameterEta(
				 distr.getDistributionParameterLink(whichDistParameter), eta));
				
		// di <- f$G.di(fv), //dv <- sum(w*di) 
		// Multiplies vector of weights with vector of global 
		//sums up all the elements of final vector in order to
		//calculate the value of global deviance(likelihood value)

		dv = w.dotProduct(distr.globalDevianceIncreament(response));
		//dv = MatrixFunctions.dotProduct(w, distr.globalDevianceIncreament(response));				
		
		//olddv <- dv+1 # the old global deviance
		olddv = dv + 1;
		
		//dldp <- f$dldp(fv) 
		dldp = distr.firstDerivative(whichDistParameter, response);

		//d2ldp2 <- f$d2ldp2(fv)
		d2ldp2 = distr.secondDerivative(whichDistParameter, response);
		
		//d2ldp2 <-  ifelse(d2ldp2 < -1e-15, d2ldp2,-1e-15)
		d2ldp2 = d2ldp2Check();
		tempArr = null;
	
		//wt <- -(d2ldp2/(dr*dr))
		wt =  wtSet();
		tempArr = null;
		
		//wv <- (eta-os)+dldp/(dr*wt)
		wv =  wvSet(offSet);
		tempArr = null;
		
		// if (family$type=="Mixed") wv <-ifelse(is.nan(wv),0,wv)
		if (distr.getTypeOfDistribution() == DistributionSettings.MIXED) {
			wv = wvCheck(wv);
			tempArr = null;
		}
		
		//iterw <- FALSE
		iterw = false;

		while (FastMath.abs(olddv - dv) > Controls.GLIM_CONV_CRIT
										 && itn < Controls.GLIM_NUM_CYCLES) {
			
//org.apache.commons.lang3.time.StopWatch watch = new org.apache.commons.lang3.time.StopWatch();
//watch.start();			
			//itn <- itn+1
			itn++;
			
			//lpold <- lp
			lpold = lp.copy();
			
			// if (any(is.na(wt))||any(is.na(wv)) ) 
			//stop("NA's in the working vector or weights for parameter ")
			if (wt.isNaN() || wv.isNaN()) {
			    System.err.println("NA's in the working vector "
			    		+ "or weights for distr parameter ");
			}
			
			// if (any(!is.finite(wt))||any(!is.finite(wv)) ) 
			//stop("Inf values in the working vector or weights for parameter") 
			if (wt.isInfinite() || wv.isInfinite()) {
			    System.err.println("Infinite values in the working "
			    		+ "vector or weights for distrparameter ");
			}
		
			//if(length(who) > 0)
			if (sMatrix != null) {
				
			   //sold <- s
			   sMatrixOld = sMatrix.copy();

			   //fit <- additive.fit(x=X,y=wv,w=wt*w,s=s,who=who,
			   //smooth.frame,maxit = bf.cyc, tol = bf.tol, trace = bf.trace
			   sMatrix = additive.fitSmoother(wv, 
					   						  wt.ebeMultiply(w),
					   						  xMatrix,
					   						  sMatrix, 
					   						  whichDistParameter);
			   
			   //lp <- if (itn==1)  
			   //fit$fitted.values else step*fit$fitted.values+(1-step)*lpold
			   //s <- if (itn==1) 
			   //fit$smooth else step*fit$smooth+(1-step)*sold 
			   if (itn == 1) {
				   lp = additive.getFittedValues();
			   } else { 
				   lp = (ArrayRealVector) 
						   		additive.getFittedValues().mapMultiply(
						   			  step).add(lpold.mapMultiply((1 - step)));
				   
				   sMatrix = (BlockRealMatrix) 
						   		sMatrix.scalarMultiply(step).add(
						   				  sMatrixOld.scalarMultiply(1 - step));
			   	}	
			   
			} else {
				
				//	fit <- lm.wfit(X,wv,wt*w,method="qr")
			   wls.newSampleData(wv, xMatrix.copy(), wt.ebeMultiply(w).copy());
			   fvLinear = (ArrayRealVector) 
					   		wls.calculateFittedValues(Controls.IS_SVD);
			   
		
			   //lp <- if (itn==1)  
			   //fit$fitted.values else step*fit$fitted.values+(1-step)*lpold
			   if (itn == 1) {
				   
				   lp = fvLinear.copy();
				} else { 
					
					lp = (ArrayRealVector) fvLinear.mapMultiply(
									  step).add(lpold.mapMultiply((1 - step)));
				}
			}
			
			// eta <- lp+os
			if (offSet != 0.0) {
				eta = MatrixFunctions.addValueToVector(lp, offSet);
			} else {
				eta = lp.copy();
			}

			//fv <- f$linkinv(eta)
			fv = makelink.linkInv(distr.getDistributionParameterLink(
													 whichDistParameter), eta);
			// replace old fitted values with new fitted values at distribution
			distr.setDistributionParameter(whichDistParameter, fv);
			
			//olddv <- dv
			olddv = dv;
			
			//di <- f$G.di(fv), //dv <- sum(w*di) 
			dv = w.dotProduct(distr.globalDevianceIncreament(response));
			//dv = MatrixFunctions.dotProduct(w, distr.globalDevianceIncreament(response));
			 
			//if (dv > olddv && itn >= 2 && auto==TRUE) 
			if (dv > olddv && itn >= 2 && Controls.AUTO_STEP) {
				//for(i in 1:5)
				for (int i = 0; i < 5; i++) {
					
					//lp <- (lp+lpold)/2
					lp = lp.add(lpold);
					lp = (ArrayRealVector) lp.mapMultiply(0.5);
				
					//eta <- lp+os
					if (offSet != 0.0) {
						eta = MatrixFunctions.addValueToVector(lp, offSet);
					} else {
						eta = lp.copy();
					}
				
					//fv <- f$linkinv(eta)
					fv = makelink.linkInv(
									distr.getDistributionParameterLink(
													whichDistParameter), eta);
					
					//replace old fitted values with new 
					//fitted values at distribution
					distr.setDistributionParameter(whichDistParameter, fv);
					
					//di <- f$G.di(fv), //dv <- sum(w*di)
					dv = w.dotProduct(distr.globalDevianceIncreament(response));
					//dv = MatrixFunctions.dotProduct(w, distr.globalDevianceIncreament(response));

					if (sMatrix != null) { 
						sMatrix = sMatrix.add(sMatrixOld);
						sMatrix = (BlockRealMatrix) sMatrix.scalarMultiply(0.5);
					}
					
					//if ((olddv-dv) > cc)
					if ((olddv - dv) > Controls.GLIM_CONV_CRIT) {
						break;
					}
				}
			}
			
			//if ((dv > olddv+gd.tol ) && itn >= 2 && iterw==FALSE) 
			//warning("The deviance has increased in an inner iteration for ",
			//"If persist, try different steps or model maybe inappropriate")
			if ((dv > olddv + Controls.GLOB_DEVIANCE_TOL) 
													   && itn >= 2 && !iterw) {
				System.err.println("Warning: The deviance has increased in an"
							+ " inner iteration for " + whichDistParameter 
							+ " " + distr.areDistributionParametersValid(
							whichDistParameter) + " If persist, try different"
							+ " steps or the model maybe inappropriate ");
				
				//iterw <-TRUE
				iterw = true;
			} 
		
			//if (is.na(!f$valid(fv))
			//stop( "fitted values in the inner iteration out of range")
			if (!distr.areDistributionParametersValid(whichDistParameter)
															   || fv.isNaN()) {
				System.err.println("fitted values in the inner "
												+ "iteration out of range");
			}
			
			//dr <- f$dr(eta), //dr <- 1/dr
			dr = MatrixFunctions.inverse(
							makelink.distParameterEta(
									distr.getDistributionParameterLink(
												whichDistParameter), eta));
		  
			//dldp <- f$dldp(fv) 
			dldp = distr.firstDerivative(whichDistParameter, response);
		
			//d2ldp2 <- f$d2ldp2(fv)
			d2ldp2 = distr.secondDerivative(whichDistParameter, response);
		  
			// d2ldp2 <-  ifelse(d2ldp2 < -1e-15, d2ldp2,-1e-15)
			d2ldp2 = d2ldp2Check();
		
			//wt <- -(d2ldp2/(dr*dr)) 
		 	 wt =  wtSet();
		
		 	 //wv <- (eta-os)+dldp/(dr*wt)
		 	 wv = wvSet(offSet);
		  
		 	 //if (family$type=="Mixed") wv <-ifelse(is.nan(wv),0,wv)
		 	 if (distr.getTypeOfDistribution() == DistributionSettings.MIXED) {
		 		 wv = wvCheck(wv);
		 	 }
		
		 	 //if(trace)
	 		 // cat("GLIM iteration ", itn, " for ",  names(formals(f$valid)),
		 	 //":Global Deviance = format(round(dv, 4))")
		 	 if (Controls.GLIM_TRACE) {
		 		 
				 System.out.println("GLIM iteration " + itn + "  for"
				      + whichDistParameter + " "
						 + distr.areDistributionParametersValid(
								              whichDistParameter)
								       + " : Global Deviance = " + dv); 
		 	 }
//watch.stop();
//System.out.println(watch.getNanoTime()/(1e+09)+"   ------");
//watch.reset();			 	 
		} // end of GlimFit While
		
		if (Controls.SMOOTHING) {
			setMatrixS(whichDistParameter, additive.getS());
		}
	}
		
    /** Set WLSMultipleLinearRegression to fit data with or without intercept.
     * @param noIntercept - boolean to specify whether the model
     *  will be fitted with or without an intercept term. */
	public final void setWLSnoIntercept(final boolean noIntercept) {
		wls.setNoIntercept(noIntercept);
	}
	
	/** Check whether vector wv has some NaN values 
	 * and if it does sets NaNs to zero.
	 * @param v = (eta-os)+dldp/(dr*wt)
	 * @return v = (eta-os)+dldp/(dr*wt) or zeros*/
	private ArrayRealVector wvCheck(final ArrayRealVector v) {
		if (v.isNaN()) {
			double[] tempA = new double[v.getDimension()];
			for (int i = 0; i < tempA.length; i++) {
				Double vD = v.getEntry(i);
		        if (vD.isNaN()) {
		        	tempA[i] = 0.0;
		        } else {
		        	tempA[i] = vD;
		        }
			}
			return new ArrayRealVector(tempA, false);
		}
		return v;
	}

	/**
	 * Calculates values of wv vector (eta-os)+dldp/(dr*wt).
	 * @param os - offset value
	 * @return wv = vector of values (eta-os)+dldp/(dr*wt)
	 */
	private ArrayRealVector wvSet(final double os) {
		tempArr = new double[eta.getDimension()];
		for (int i = 0; i < tempArr.length; i++) {
			tempArr[i] = (eta.getEntry(i) - os) + dldp.getEntry(i)
										/ (dr.getEntry(i) * wt.getEntry(i));
		}
		return new ArrayRealVector(tempArr, false);  
	}

	/**
	 * Calculates values of wt vector wt = -(d2ldp2/(dr*dr)).
	 * @return wt = -(d2ldp2/(dr*dr))
	 */
	private ArrayRealVector wtSet() {
		tempArr = new double[d2ldp2.getDimension()];
		for (int i = 0; i < tempArr.length; i++) {
			tempArr[i] = -(d2ldp2.getEntry(i) 
					/ (dr.getEntry(i) * dr.getEntry(i)));
		}
		return new ArrayRealVector(tempArr, false); 
	}
	
	/**
	 * Checks whether the values of the second derrivative greater 
	 * than -1e-15d and if they are, sets these values = -1e-15d.
	 * @return vector of second derivative values with respect to 
	 * the fitted distribution parameter
	 */
	private ArrayRealVector d2ldp2Check() {
		//ifelse(d2ldp2 < -1e-15, d2ldp2,-1e-15)
		double value = -1e-15d;
		tempArr = new double[d2ldp2.getDimension()];
		for (int i = 0; i < tempArr.length; i++) {
			if (d2ldp2.getEntry(i) > value) {
				tempArr[i] = value; 
			} else {
				tempArr[i] = d2ldp2.getEntry(i);
			}
		}
		return new ArrayRealVector(tempArr, false); 
	}
	
	/**
	 * Set smother matrix of the currently fitting 
	 * (whichDistParameter) distribution parameter. 
	 * @param s - smother matrix
	 * @param whichDistParameter - fitting distribution parameter
	 */
	public final void setMatrixS(final int whichDistParameter,
											final BlockRealMatrix s) {
		sMatrices.put(whichDistParameter, s);
	}
		
}
