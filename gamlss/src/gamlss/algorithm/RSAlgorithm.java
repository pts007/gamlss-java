/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;

import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.smoothing.PB;
import gamlss.smoothing.RW;
import gamlss.utilities.Controls;
import gamlss.utilities.MatrixFunctions;
import gamlss.utilities.oi.CSVFileReader;
import gamlss.utilities.oi.ConnectionToR;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */

public class RSAlgorithm {

	/** Identifier of currently fitting distribution parameter. */
	private int whichDistParameter;
	/** RS algorithm convergence boolean. */
	private boolean conv;
	/** Number of RS Algorithm iterations performed. */
	private double iter;
	/** Global deviance. */
	private double gDev;
	/** Previously stored value of the global deviance. */
	private double gDevOld;
	/** Object of Gamlss fitting algorithm. */
	private GlimFit glimfit;

	/**
	 * this is to emulate the RSAlgorithm iterative algorithm.
	 * @param distr - object of the fitted distribution belonging
	 *  to the gamlss family
	 * @param response  -  vector of response variable values
	 * @param designMatrices - design matrices for 
	 * each of the distribution parameters
	 * @param smoothMatrices - smoother matrices for each 
	 * of the distribution parameters
	 * @param w - vector of the weight values
	 *  the original data or not
	 *  distribution parameters
	 */
	public RSAlgorithm(final GAMLSSFamilyDistribution distr,
					   final ArrayRealVector response, 
					   final Hashtable<Integer, BlockRealMatrix> designMatrices,
					   final HashMap<Integer, BlockRealMatrix> smoothMatrices,
					   final ArrayRealVector w) {
		
		glimfit = new GlimFit(distr, 
							  response, 
							  designMatrices, 
							  smoothMatrices, 
							  w);
		
		if (Controls.SMOOTHING) {
			for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
				if (smoothMatrices.get(i) != null) { 
						// Create smoother matrices of zeros
					glimfit.setMatrixS(i, new BlockRealMatrix(
								smoothMatrices.get(i).getRowDimension(),
								smoothMatrices.get(i).getColumnDimension()));
				} else {
					glimfit.setMatrixS(i, null);
				}
			}
		} else {
			for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
				glimfit.setMatrixS(i, null);
			}
		}
	}
	
	/**
	 *  The Rigby and Stasinopoulos fitting algorithm. 
	 *  Uses WLSMultipleLinearRegression to calculate the fitting
	 *   coefficents to define the fitted values of the distribution
	 *    parameters, also updating them and storing them in the
	 *     supplied distribution class (distr).
	 *  
	 * @param distr - object of the fitted distribution belonging
	 *  to the gamlss family
	 * @param response  -  vector of response variable values
	 * @param w - vector of the weight values
	 *  the original data or not
	 * @param designMatrices - design matrices for 
	 * each of the distribution parameters
	 * @param smoothMatrices - smoother matrices for each 
	 * of the distribution parameters
	 *  distribution parameters 
	 */ 
	public void  functionRS(final GAMLSSFamilyDistribution distr,
								  final ArrayRealVector response,
								  final ArrayRealVector w) {
		
		//iter <- control$iter
		iter = Controls.ITERATIONS;
		
		//conv <- FALSE
		conv = false;

		//G.dev <- sum(w*G.dev.incr)
		gDev = w.dotProduct(distr.globalDevianceIncreament(response));
		
		//G.dev.old <- G.dev+1
		gDevOld = gDev + 1;
		
        //while ( abs(G.dev.old-G.dev) > c.crit && iter < n.cyc ){
		while (FastMath.abs(gDevOld - gDev) 
						> Controls.GAMLSS_CONV_CRIT 
									  && iter < Controls.GAMLSS_NUM_CYCLES) {

			
			//if ("mu"%in%names(family$parameters)){
			//if ("sigma"%in%names(family$parameters)){
			//if ("nu"%in%names(family$parameters)){
			//if ("tau"%in%names(family$parameters)){
			for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
				switch (i) {
		        case DistributionSettings.MU:
		        	 whichDistParameter = DistributionSettings.MU;
		           break;
		        case DistributionSettings.SIGMA:
		        	 whichDistParameter = DistributionSettings.SIGMA;
		          break;
		        case DistributionSettings.NU:
		        	 whichDistParameter = DistributionSettings.NU;
		          break;
		        case DistributionSettings.TAU:
		        	 whichDistParameter = DistributionSettings.TAU;
		          break;
		        default: 
					System.err.println(" Distribution parameter "
														+ "is not allowed");
				}
				
				//if  (family$parameter$mu==TRUE & mu.fix==FALSE){
				glimfit.setWLSnoIntercept(
								Controls.NO_INTERCEPT[whichDistParameter - 1]);
				
				// mu.fit <<- glim.fit(f = mu.object, X = mu.X, y = y, w = w,
				//fv = mu, os = mu.offset, step = mu.step,control = i.control,
				//gd.tol = gd.tol,auto = autostep)
				glimfit.glimFitFunctionRS(whichDistParameter);
						
			}
			
			//G.dev.old <- G.dev
			gDevOld = gDev;
		
			//G.dev <- sum(w*G.dev.incr)
			gDev =  w.dotProduct(distr.globalDevianceIncreament(response));
			
			//iter <- iter+1
			iter = iter + 1;
			
			//fiter <<- iter

            // if(trace) cat("GAMLSS-RS iteration ", iter, ": Global Deviance
			//= ", format(round(G.dev, 4)), " \n", sep = "")
			if (Controls.GAMLSS_TRACE) {
				System.out.println("GAMLSS-RS iteration "
									 + iter + " : Global Deviance = " + gDev);
			}
			
            //if (G.dev > (G.dev.old+gd.tol) && iter >1 ) stop(paste("The
			//global deviance is increasing", "\n", "Try different 
			//steps for the parameters or the model maybe inappropriate"))
			if (gDev > (gDevOld + Controls.GLOB_DEVIANCE_TOL) && iter > 1) {
				System.err.println("The global deviance is increasing, "
						+ "Try different steps for the parameters or "
										+ "the model maybe inappropriate");
				break;
			}

			//if ( abs(G.dev.old-G.dev) < c.crit ) conv <- TRUE else FALSE
			 if (FastMath.abs(gDevOld - gDev) < Controls.GAMLSS_CONV_CRIT) {
				 conv = true;
			 }
		 
		}
			 
		//if (!conv && no.warn )   
		if (!conv) {	
			//warning("Algorithm RS has not yet converged")
			System.err.println("Algorithm RS has not yet converged");
		}
	}
}
