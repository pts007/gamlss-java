/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.utilities.oi;


import java.io.File;

import javax.swing.JOptionPane;

import org.rosuda.REngine.REXP;
import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;
import org.rosuda.REngine.Rserve.RConnection;
import org.rosuda.REngine.Rserve.RserveException;

/**Top-level class to connect ACEGES with R for statistical modelling
 * @author Vlasios Voudouris
 * Created: October, 2010
 */
public class ConnectionToR extends StartRserve 
{	
	private RConnection rEngine=null;	
	
	public ConnectionToR() 
	{
		checkLocalRserve();
		try {			
			 rEngine = new RConnection();
		} catch (RserveException e) {
			e.printStackTrace();
		} 				
	}
 	
  /**
   * Evaluates the commend and return the results as REXP
   * @param cmd
   * @return REXP
   */
  public REXP runEval(String cmd)
  {
	  REXP rEXP=null;
	  try {
		  rEXP= rEngine.eval(cmd);
	} catch (REngineException e) {
		JOptionPane.showMessageDialog(null, cmd);
		e.printStackTrace();
		System.exit(0);
	} 
	return rEXP;
  }
  
  /**
   * Evaluates the commend and return the results as doubles
   * @param cmd
   * @return double[]
   */
  public double[] runEvalDoubles(String cmd)
  {
	double[] rEXP=null;	
	    try {
			rEXP= rEngine.eval(cmd).asDoubles();
		} catch (REXPMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (REngineException e) {
		  JOptionPane.showMessageDialog(null, cmd);
		  e.printStackTrace();
		  System.exit(0);
	    } 
	return rEXP;
  }
  
  
  
  /**
   * Evaluates the commend and return the results as double
   * @param cmd
   * @return double
   */
  public double runEvalDouble(String cmd)
  {
	  double rEXP=0.0;	
	    try {
			rEXP= rEngine.eval(cmd).asDouble();
		} catch (REXPMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (REngineException e) {
		  //JOptionPane.showMessageDialog(null, "To run the full function of ACEGES, you need R");
		  e.printStackTrace();
		  //System.exit(0);
	    } 
	return rEXP;
  }
  
  /**
   * Evaluates the commend and return the results as String
   * @param cmd
   * @return REXP
   */
  public String runEvalString(String cmd)
  {
	  String rEXP=null;
	  try {
		  try {
			rEXP= rEngine.eval(cmd).asString();
		} catch (REXPMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	} catch (REngineException e) {
		//JOptionPane.showMessageDialog(null, "To run the full function of ACEGES, you need R");
		e.printStackTrace();
		//System.exit(0);
	} 
	return rEXP;
  }
  
  
  /**
   * Evaluates the command and returns nothing. 
   * @param cmd
   */
  public void voidEval(String cmd)
  {
	  try {
		  rEngine.voidEval(cmd);
	} catch (REngineException e) {
		//JOptionPane.showMessageDialog(null, cmd);
		e.printStackTrace();
		//System.exit(0);
	}
  }
  
  /**
   * 
   * @param variable: This is the R name of the variable 
   * @param data: This is the data asinged to the variable 
   */
  public void assingVar(String variable, double[] data)
  {
	  REXP rEXP=null;
	  try {
		  rEngine.assign(variable, data);
	} catch (REngineException e) {
		e.printStackTrace();
	}
  }
   
  /**
   * This closes the RServe
   */
  public void shutDown()
  {
	  try {
		rEngine.shutdown();
	} catch (RserveException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }

  /**
   * Sources a script file (usually to be used by agents)
   * @param fileName is the relative path of the script. 
   */
  public void sourceRScript(String fileName)
  {
	  File file = new File(fileName);
	  String cmd= "source('"+file.getAbsolutePath()+"')";
	  this.runEval(cmd);	 	  
  }
  
  /**
   * read a data to be used by the ACEGES model or agents. 
   * @param fileName is the relative path of the script. 
   *  @param rParameter to hold the data: data = read.csv(path2, header = T)
   */
  public void readCSVFile(String rParameter, String fileName)
  {
	  File file = new File(fileName);
	  String cmd= rParameter + "<- read.csv('"+ file.getAbsolutePath()+"', header=T)";
	  
	  this.runEval(cmd);	 	  
  }
	
	/** just a demo main method which starts Rserve and shuts it down again */
    public static void main(String[] args) 
    {   	    	
    	ConnectionToR test = new ConnectionToR();
    	test.shutDown();
    	System.exit(0);
    }
    
	
}
