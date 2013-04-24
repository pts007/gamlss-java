/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/

package gamlss.algorithm;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.PointValuePair;
import org.apache.commons.math3.optimization.direct.BaseAbstractMultivariateOptimizer;
import org.apache.commons.math3.optimization.direct.MultivariateFunctionMappingAdapter;

/**
 * Performs global optimisation of the GAMLSS model. This is an alternative to RS and CG algorithm in GAMLSSS
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */
public class GlobalOptimGAMLSS 
{
	private MultivariateFunctionMappingAdapter objFunction;	
	
	/**
	 * performs global optimisation of the log-likelihood 
	 * @param objFunction the log-likelihood function 
	 * @param lower the lower limits based upon the permissible values of the distribution parameters
	 * @param upper the upper limits based upon the permissible values of the distribution parameters
	 */
	public GlobalOptimGAMLSS (MultivariateFunction objFunction,  final double[] lower, final double[] upper)
	{
		this.objFunction= new MultivariateFunctionMappingAdapter(objFunction,lower,upper);
	}
	
	/**
	 * optimises the objective function
	 * @param optim the optimizer such as CMAESOptimizer or BOBYQAOptimizer
	 * @param startValues the starting values of the optimisation 
	 * @return the point/value pair giving the optimal value for objective function. 
	 */
	public PointValuePair optimazer(BaseAbstractMultivariateOptimizer<MultivariateFunctionMappingAdapter> optim, double[] startValues)
	{
		PointValuePair results = null;
		 try 
		 {
			 results= optim.optimize(100000, this.objFunction, GoalType.MINIMIZE, startValues);	
		 }
		 catch (MaxCountExceededException e) 
		 {
			 System.err.println("more optimisation steps are needed - best results so far are returned");
		 }
		 return results;
	}

	/**
	 * 
	 * @return the log-likelihood function 
	 */
	public MultivariateFunctionMappingAdapter getObjFunction() {
		return objFunction;
	}

	/**
	 * 
	 * @param objFunction the log-likelihood function 
	 */
	public void setObjFunction(MultivariateFunctionMappingAdapter objFunction) {
		this.objFunction = objFunction;
	}

}


