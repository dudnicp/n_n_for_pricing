using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.GradientAdjustment
{
    public static class GradientAdjustmentStrategyFactory
    {
        public static IGradientAdjustmentStrategy Build(IGradientAdjustmentParameters parameters)
        {
            switch (parameters)
            {
                case AdamParameters adamParameters:
                    return new AdamStrategy(adamParameters);
                case FixedLearningRateParameters fixedLearningRateParameters:
                    return new FixedLearningRateStrategy(fixedLearningRateParameters);
                case MomentumParameters momentumParameters:
                    return new MomentumStrategy(momentumParameters);
                case NesterovParameters nesterovParameters:
                    return new NesterovStrategy(nesterovParameters);
                default:
                    throw new InvalidOperationException("Unknown parameters type: " + parameters.GetType());
            }
        }
    }
}
