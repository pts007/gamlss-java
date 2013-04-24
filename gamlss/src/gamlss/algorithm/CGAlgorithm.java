/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;


import java.util.Hashtable;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.util.FastMath;

import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.WLSMultipleLinearRegression;

/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */

public class CGAlgorithm {
	
	
	private ArrayRealVector dr;
	private ArrayRealVector eta;
	private ArrayRealVector etaOld;
	private ArrayRealVector wMuSigma;
	private ArrayRealVector wMuNu;
	private ArrayRealVector wMuTau; 
	private ArrayRealVector w;
	private ArrayRealVector z;
	private ArrayRealVector wSigmaNu; 
	private ArrayRealVector wSigmaTau;
	private ArrayRealVector u2MuSigma;
	

	private int nCyc;
	Hashtable<Integer, ArrayRealVector> bettas = new Hashtable<Integer, ArrayRealVector>();
	
	private GlimFitCG glimfitCG; 
	private MakeLinkFunction makelink;
	
	Hashtable<String, ArrayRealVector> cgData = new Hashtable<String, ArrayRealVector>();
	/** identifier of distribution parameter */
	private int whichDistParameter;
	

	

	
	public CGAlgorithm(int size) {
	
		nCyc = Controls.GAMLSS_NUM_CYCLES;
		glimfitCG = new GlimFitCG(Controls.COPY_ORIGINAL, size);
		
		makelink = new MakeLinkFunction();
		
	    //w.mu.sigma<-w.mu.nu<-w.mu.tau<-w.sigma.nu<-w.sigma.tau<-w.nu.tau<-rep(0,N)            
        //eta.mu<-eta.old.mu<-eta.sigma<-eta.old.sigma<-eta.nu<-eta.old.nu<-rep(0,N)  
        //eta.old.tau<-eta.tau<-rep(0,N) #???
		eta = new ArrayRealVector(size);
		etaOld = new ArrayRealVector(size);
		wMuNu = new ArrayRealVector(size);
		wMuTau = new ArrayRealVector(size);
//		w = new ArrayRealVector(size);
		wSigmaNu = new ArrayRealVector(size);
		wSigmaTau = new ArrayRealVector(size);
		
		 for (int i=1; i<5; i++)
		 {
			 cgData.put("eta"+i, eta);
			 cgData.put("etaOld"+i, etaOld); 
		 }
		 
		 cgData.put("wMuNu", wMuNu);
		 cgData.put("wMuTau", wMuTau);
		 cgData.put("wSigmaNu", wSigmaNu);
		 cgData.put("wSigmaTau", wSigmaTau);
	}
	
	
	public Hashtable<Integer, ArrayRealVector> CGfunction (GAMLSSFamilyDistribution distr, ArrayRealVector response, 
															Hashtable<Integer, BlockRealMatrix> X, ArrayRealVector weights 
																		){
        

        //iter <- control$iter
        double iter = Controls.ITERATIONS;
        
        //conv <- FALSE
        boolean conv = false;
        

        //G.dev.incr <- eval(G.dev.expr)                  
        //G.dev <- sum(w*G.dev.incr)
        double gDev = weights.dotProduct(distr.globalDevianceIncreament(response));
        
        //G.dev.old <- G.dev+1  
        double gDevOld = gDev+1;

        //while ( abs(G.dev.old-G.dev) > c.crit && iter < n.cyc )
        while (FastMath.abs(gDevOld - gDev) > Controls.GAMLSS_CONV_CRIT && iter < nCyc)
        {
        	
        	
        	
			for (int i = 1; i < distr.getNumberOfDistribtionParameters()+1; i++ ){
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
			}
        	
        	
    		//eta.mu <- eta.old.mu <- family$mu.linkfun(mu)    
    		etaOld = makelink.link(distr.getDistributionParameterLink(whichDistParameter), 
    																	distr.getDistributionParameter(whichDistParameter));
    		cgData.put("etaOld"+whichDistParameter, etaOld);
    		
    		eta = etaOld;
    		cgData.put("eta"+whichDistParameter, eta);
    		
    		//u.mu <- mu.object$dldp(mu=mu)
    		ArrayRealVector u = distr.firstDerivative(whichDistParameter, response);
    		
    		//u2.mu <- mu.object$d2ldp2(mu=mu)
    		ArrayRealVector u2 = distr.secondDerivative(whichDistParameter, response);
    		
    		if (whichDistParameter == DistributionSettings.SIGMA)
    		{
    			u2MuSigma= distr.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.SIGMA, response);
    			
    		}
    		
    		//dr.mu <- family$mu.dr(eta.mu)
    		dr = makelink.distParameterEta(distr.getDistributionParameterLink(whichDistParameter), eta);
    		
    		//dr.mu <- 1/dr.mu
    		dr = drInverse(dr);
    		cgData.put("dr"+whichDistParameter, dr);
    		//who.mu <- mu.object$who
    		//smooth.frame.mu <- mu.object$smooth.frame
            //s.mu <- if(first.iter) mu.object$smooth else s.mu       
    		
    		//w.mu <- -u2.mu/(dr.mu*dr.mu)
           	w =  wtSet(u2, dr);
           	cgData.put("w"+whichDistParameter, w);
           	
           	if (whichDistParameter == DistributionSettings.SIGMA)
           	{
           		wMuSigma = wMUSIGMAset(u2MuSigma, cgData.get("dr"+DistributionSettings.MU), dr);
           		cgData.put("wMuSigma", wMuSigma);
           	}
    		//z.mu <- (eta.old.mu-mu.offset)+mu.step*u.mu/(dr.mu*w.mu)
           	z = wvSet(etaOld, Controls.STEP[whichDistParameter-1], Controls.OFFSET[whichDistParameter-1], u, dr, w);
           	cgData.put("z"+whichDistParameter, z);  
        }
        	
        	
        	
        	
        	
        	
        	
        	
        		
			for (int i = 1; i < distr.getNumberOfDistribtionParameters()+1; i++ ){
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
			}
				
				
			//if  (family$parameter$mu==TRUE & mu.fix==FALSE){
			glimfitCG.setWLSnoIntercept(Controls.NO_INTERCEPT[whichDistParameter-1]);
 
            
			bettas.put(whichDistParameter, glimfitCG.glimFitFunctionCG(distr, response,  X,  
						distr.getDistributionParameter(whichDistParameter), weights,  whichDistParameter, 
						Controls.STEP[whichDistParameter-1], Controls.OFFSET[whichDistParameter-1], gDev, cgData, makelink));
            
            
            
			}
        	
			gDevOld = gDev;
			
        	//G.dev.incr <- eval(G.dev.expr)   
        	//G.dev <- sum(w*G.dev.incr)
        	gDev = weights.dotProduct(distr.globalDevianceIncreament(response));
                
        			//if (G.dev < G.dev.old) break
        		//	if (gDev < gDevOld)
        		//	{ 
        		//		break;
        		//	}
        		
        	
        
            //iter <- iter+1  
            iter = iter+1;
  
            //if(trace)
            if(Controls.GAMLSS_TRACE)
            {
            	//cat("GAMLSS-CG iteration ", iter, ": Global Deviance = ", format(round(G.dev, 4)), " \n", sep = "")
        		System.out.println("GAMLSS-CG iteration "+iter+" : Global Deviance = "+ gDev);
            }
            //if (G.dev > (G.dev.old+gd.tol) && iter >1 )
            if(gDev > (gDevOld + Controls.GLOB_DEVIANCE_TOL) && iter > 1)
        	{
            	//stop(paste("The global deviance is increasing in CG-algorithm ", "\n",  "Try different steps for the parameters or the model maybe inappropriate"))
            	System.err.println("The global deviance is increasing in CG-algorithm, Try different steps for the parameters or the model maybe inappropriate");
            	break;
        	}
            
       
	
       //if ( abs(G.dev.old-G.dev) < c.crit ) conv <- TRUE else FALSE 
        if (FastMath.abs(gDevOld - gDev) < Controls.GAMLSS_CONV_CRIT){
        	
        	conv = true;
        }
        else
        {
        	conv = false;
        }
        // if (!conv)   
        if (!conv && iter == nCyc)
        {
        	//warning("Algorithm CG has not yet converged");
        	System.out.println("Algorithm CG has not yet converged");
        }
        }
        return bettas;
	}
	
	public void setnCyc(int nCyc) {
		this.nCyc = nCyc;
	}
	
	//--------------------------------------------------------------------------------------------------------
   	/**
   	 *  Calculates inverse of dr vector values (1/dr)
   	 * @param dr - vector of 1/(link function of the linear pridictoor) values
   	 * @return 1/dr
   	 */
   	private ArrayRealVector drInverse(ArrayRealVector dr){
    			double[] out = new double[dr.getDimension()];
    			for (int i=0; i<out.length; i++)
    			{
    				out[i] = 1/dr.getEntry(i);
    			}	
    			return new ArrayRealVector(out,false);
    	}
  	
   	
  //--------------------------------------------------------------------------------------------------------		
  	/**
  	 * Calculates values of wt vector
  	 * @param d2ldp2 - vector of second derivative values with respect to the fitted distribution parameter
  	 * @param dr - vector of 1/(link function of the linear pridictoor) values
  	 * @return wt = -(d2ldp2/(dr*dr))
  	 */
  	private ArrayRealVector wtSet(ArrayRealVector d2ldp2, ArrayRealVector dr){
  		double[] out = new double[d2ldp2.getDimension()];
  		for (int i=0; i<out.length; i++)
  		{
  			out[i] = -(d2ldp2.getEntry(i)/(dr.getEntry(i)*dr.getEntry(i)));
  		}
  		return new ArrayRealVector(out,false); 
  	} 	
   	
  	
    //--------------------------------------------------------------------------------------------------------	
  	/**
  	 * Calculates values of wv vector
  	 * @param eta - vector of linear predictor values
  	 * @param os - offset value
  	 * @param dldp - vector of the first derivative values with respect to the fitted distribution parameter
  	 * @param dr = vector of 1/(link function of the linear pridictoor) values
  	 * @param wt = vectr f -(second derivative values with respect to the fitted distribution parameter/(dr*dr)) values
  	 * @return wv = vector of values (eta-os)+dldp/(dr*wt)
  	 */
  	private ArrayRealVector wvSet(ArrayRealVector etaOldMU, double muStep, double os, ArrayRealVector dldp, ArrayRealVector dr, ArrayRealVector wt){
  		double[] out = new double[etaOldMU.getDimension()];
  		for (int i=0; i<out.length; i++)
  		{
  			out[i] = (etaOldMU.getEntry(i)-os)+muStep*dldp.getEntry(i)/(dr.getEntry(i)*wt.getEntry(i));
  		}
  		return new ArrayRealVector(out,false);  
  	}
  
  //--------------------------------------------------------------------------------------------------------		
  	/**
  	 * Calculates values of wt vector
  	 * @param d2ldp2 - vector of second derivative values with respect to the fitted distribution parameter
  	 * @param dr - vector of 1/(link function of the linear pridictoor) values
  	 * @return wt = -(d2ldp2/(dr*dr))
  	 */
  	private ArrayRealVector wMUSIGMAset(ArrayRealVector u2MUSIGMA, ArrayRealVector drMU, ArrayRealVector drSIGMA){
  		double[] out = new double[u2MUSIGMA.getDimension()];
  		for (int i=0; i<out.length; i++)
  		{
  			out[i] = -(u2MUSIGMA.getEntry(i)/(drMU.getEntry(i)*drSIGMA.getEntry(i)));
  		}
  		return new ArrayRealVector(out,false); 
  	}
  	
}
