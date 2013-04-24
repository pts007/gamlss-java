/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;

import gamlss.distributions.DistributionSettings;
import gamlss.distributions.GAMLSSFamilyDistribution;
import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.WLSMultipleLinearRegression;

import java.util.Hashtable;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.util.FastMath;

public class GlimFitCG {
	
	private int nCyc;
//	Hashtable<Integer, ArrayRealVector> bettas = new Hashtable<Integer, ArrayRealVector>();

//	private MakeLinkFunction makelink;
	private WLSMultipleLinearRegression wls;
	
//	Hashtable<String, ArrayRealVector> cgData = new Hashtable<String, ArrayRealVector>();
	
	/** identifier of distribution parameter */

	private ArrayRealVector adj;	


	
	public GlimFitCG(boolean copyOriginal, int size){
		
		
		wls = new WLSMultipleLinearRegression(copyOriginal);
		

		adj = new ArrayRealVector(size);
		

	}
		
	
	
	public ArrayRealVector glimFitFunctionCG(GAMLSSFamilyDistribution distr, ArrayRealVector response,
			Hashtable<Integer, BlockRealMatrix> X, ArrayRealVector fv, ArrayRealVector weights,
						int whichDistParameter, double step, double offSet, double globalDeviance, Hashtable<String, ArrayRealVector> cgData, MakeLinkFunction makelink){
	
		int itn = 0;
		double[] temp = new double[0];
		
            	
	             	
	
       	//G.dev.in <- G.dev+1
     //  	double gDevIn = globalDeviance +1;

       	//i.G.dev <- G.dev
       	double gDev = globalDeviance;
       	
       	double gDevOld = globalDeviance+1;

       	//first.iter <- FALSE  
       	boolean firstIter = false;
       	
        //while ( abs(G.dev.in -i.G.dev) > i.c.crit && i.iter < i.n.cyc )
        while (FastMath.abs(gDevOld - gDev) > Controls.GLIM_CONV_CRIT && itn < Controls.GLIM_NUM_CYCLES)
        {
        	
    		//i.iter <- i.iter+1
    		itn = itn+1;
    		
			//for (int i = 1; i < distr.getNumberOfDistribtionParameters()+1; i++ ){
				switch (whichDistParameter) {
		        case DistributionSettings.MU:
		        	// whichDistParameter = DistributionSettings.MU;
		        	 
	            		//adj.mu <- -(w.mu.sigma*(eta.sigma-eta.old.sigma)+w.mu.nu*(eta.nu-eta.old.nu)+w.mu.tau*(eta.tau-eta.old.tau))/w.mu 
	            		adj = setAdj(cgData.get("wMuSigma"), 
	            				     cgData.get("eta"+DistributionSettings.SIGMA), 
	            				     cgData.get("etaOld"+DistributionSettings.SIGMA), 
	            				     cgData.get("wMuNu"),
	            				     cgData.get("eta"+DistributionSettings.NU),
	            				     cgData.get("etaOld"+DistributionSettings.NU),
	            				     cgData.get("wMuTau"), 
	            				     cgData.get("eta"+DistributionSettings.TAU),
	            				     cgData.get("etaOld"+DistributionSettings.TAU), 
	            				     cgData.get("w"+DistributionSettings.MU));
	            		
	            		cgData.put("adj"+DistributionSettings.MU, adj);
		        	 
		           break;
		        case DistributionSettings.SIGMA:
		        	// whichDistParameter = DistributionSettings.SIGMA;
		        	 
	            		// adj.sigma <- -(w.mu.sigma*(eta.mu-eta.old.mu)+ w.sigma.nu*(eta.nu-eta.old.nu)+w.sigma.tau*(eta.tau-eta.old.tau))/w.sigma 
	           		 adj = setAdj(  cgData.get("wMuSigma"),
	           				 	  	cgData.get("eta"+DistributionSettings.MU),
	           				 	  	cgData.get("etaOld"+DistributionSettings.MU),
	           				 	  	cgData.get("wSigmaNu"),
	           				 	  	cgData.get("eta"+DistributionSettings.NU),
	           				 	  	cgData.get("etaOld"+DistributionSettings.NU),
	           				 	  	cgData.get("wSigmaTau"),
	           				 	  	cgData.get("eta"+DistributionSettings.TAU),
	           				 	  	cgData.get("etaOld"+DistributionSettings.TAU),
	           				 	  	cgData.get("w"+DistributionSettings.SIGMA));
	           		 
	           		 cgData.put("adj"+DistributionSettings.SIGMA, adj);
		        	 
		          break;
		        case DistributionSettings.NU:
		        	// whichDistParameter = DistributionSettings.NU;
		          break;
		        case DistributionSettings.TAU:
		        	// whichDistParameter = DistributionSettings.TAU;
		          break;
				}            		
        		
				
        		//wv.mu  <- z.mu+adj.mu
        		ArrayRealVector wv = setWV(cgData.get("z"+whichDistParameter), cgData.get("adj"+whichDistParameter));
        		
                //mu.fit <<- lm.wfit(x=mu.X,y=wv.mu,w=w.mu*w,method="qr") 
        		wls.newSampleData(wv,  X.get(whichDistParameter).copy(), cgData.get("w"+whichDistParameter).ebeMultiply(weights));	
        		ArrayRealVector fit = (ArrayRealVector) wls.calculateFittedValues(false);
        		
        		//System.out.println(wls.calculateBeta());
   //     		bettas.put(whichDistParameter, (ArrayRealVector) wls.calculateBeta());

        		//mu.fit$eta <<- eta.mu <- mu.fit$fitted.values+mu.offset
        		temp = new double[fit.getDimension()];
        		for (int j=0; j<temp.length; j++)
        		{
        			temp[j] = fit.getEntry(j) + offSet;
        		}
        		//eta = new ArrayRealVector(temp,false);
        		cgData.put("eta"+whichDistParameter, new ArrayRealVector(temp,false));
        		temp = null;
        		
        		//mu.fit$fv <<-    mu <<- mu.object$linkinv(eta.mu)
				
        		ArrayRealVector dParam =makelink.linkInv(distr.getDistributionParameterLink(whichDistParameter), cgData.get("eta"+whichDistParameter));
        		
        		distr.setDistributionParameter(whichDistParameter, dParam);
        		
        		//mu.fit$wv <<- wv.mu
        		//mu.fit$wt <<- w.mu  
        
			
        		//G.dev.in <- i.G.dev
        		gDevOld = gDev;
        
        		//G.dev.incr <- eval(G.dev.expr) 
        		//i.G.dev <- sum(w*G.dev.incr)
        		gDev = weights.dotProduct(distr.globalDevianceIncreament(response));
         
               	if (gDev > gDevOld && itn >= 2 && Controls.AUTO_STEP)
        		{
            		for(int i=0; i<5; i++){
        		
      
     
        				//eta.mu <- (eta.mu+eta.old.mu)/2
            				ArrayRealVector etaM = etaMean(cgData.get("eta"+whichDistParameter), cgData.get("etaOld"+whichDistParameter));
        				cgData.put("eta"+whichDistParameter, etaM);		
        				//mu <<- mu.object$linkinv(eta.mu) 
    					
        				distr.setDistributionParameter(whichDistParameter, 
    							makelink.linkInv(distr.getDistributionParameterLink(whichDistParameter), etaM));
    				
    					//if(length(who.mu) > 0) s.mu <- (s.mu+s.mu.old)/2 
        				//	}
            		}     
        		}
        		

        		//if(i.trace)
        		if(Controls.GLIM_TRACE)
        		{
        			//cat("CG inner iteration ", iter, ": Global Deviance = ",format(round(i.G.dev, 4)), " \n", sep = "")           
        			System.err.println("CG inner iteration "+itn+" : Global Deviance = "+gDev);
        		}
        		//if (i.G.dev > (G.dev.in+gd.tol) && iter >1 )
        		if(gDev > (gDevOld + Controls.GLOB_DEVIANCE_TOL) && itn >1)
        		{
        	 
        			//stop(paste("The global deviance is increasing in the inner CG loop", "\n","Try different steps for the parameters or the model maybe inappropriate"))
        			System.out.println("The global deviance is increasing in the inner CG loop, try different steps for the parameters or the model maybe inappropriate");
        			break;
        		}
        	}
        
        	
        
        //G.dev.old <- G.dev
   //     	gDevOld = gDev;
        
        	//G.dev.incr <- eval(G.dev.expr)   
        	//G.dev <- sum(w*G.dev.incr)
      //  	gDev = weights.dotProduct(distr.globalDevianceIncreament(response));
    
        	//if (G.dev > G.dev.old && iter >= 2 && autostep == TRUE)
 
        	return (ArrayRealVector) wls.calculateBeta();
		}



	
	
//--------------------------------------------------------------------------------------------------------		
  /** @param noIntercept - boolean to specify whether the model will be estimated with or without an intercept term */
  public void setWLSnoIntercept(boolean noIntercept){
	  wls.setNoIntercept(noIntercept);
  }
    		
	

//--------------------------------------------------------------------------------------------------------
	private ArrayRealVector setWV(ArrayRealVector in1, ArrayRealVector in2){
		
		double[] out = new double[in1.getDimension()];
		for (int i=0; i<out.length; i++)
		{
			out[i] = in1.getEntry(i) + in2.getEntry(i);
		}
		return new ArrayRealVector(out,false);
	}
	
//--------------------------------------------------------------------------------------------------------	
	private ArrayRealVector setAdj(ArrayRealVector in1, ArrayRealVector in2, ArrayRealVector in3,
										ArrayRealVector in4, ArrayRealVector in5, ArrayRealVector in6,
										ArrayRealVector in7,ArrayRealVector in8, ArrayRealVector in9,
										ArrayRealVector in10){

		double[] out = new double[in10.getDimension()];
		for (int i=0; i<out.length; i++)
		{
			out[i] = -(in1.getEntry(i)*(in2.getEntry(i) - in3.getEntry(i)) + 
					  		in4.getEntry(i)*(in5.getEntry(i) - in6.getEntry(i)) + 
					  			in7.getEntry(i)*(in8.getEntry(i) - in9.getEntry(i)))/in10.getEntry(i);
		}
		return new ArrayRealVector(out,false);
	}
        

//--------------------------------------------------------------------------------------------------------	
	private ArrayRealVector etaMean(ArrayRealVector etaMU, ArrayRealVector etaOldMU){
		
		double[] out = new double[etaMU.getDimension()];
		for (int i=0; i<out.length; i++)
		{
			out[i] = (etaMU.getEntry(i) + etaOldMU.getEntry(i))/2;
		}
		return new ArrayRealVector(out,false);
	}
	
	
	

        

  	



	

}
