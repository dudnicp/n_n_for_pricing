using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.GradientAdjustment
{
    public static class GradientAdjustmentParametersFactory
    {
        public static IGradientAdjustmentParameters Build(GradientAdjustmentType type)
        {
            switch (type)
            {
                case GradientAdjustmentType.Adam:
                    return new AdamParameters();
                case GradientAdjustmentType.FixedLearningRate:
                    return new FixedLearningRateParameters();
                case GradientAdjustmentType.Momentum:
                    return new MomentumParameters();
                case GradientAdjustmentType.Nesterov:
                    return new NesterovParameters();
                default:
                    throw new InvalidOperationException("Unknown parameters type: " + type.ToString());
            }
        }
    }
}
