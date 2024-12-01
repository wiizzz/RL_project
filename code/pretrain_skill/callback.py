from stable_baselines3.common.callbacks import BaseCallback


def get_low_act(data, threshold=0.2):
    """Computes the proportion of activations that have value close to zero."""
    low_activation = ((-threshold <= data) & (data <= threshold))
    return np.count_nonzero(low_activation) / np.size(low_activation)


# Callback for periodic logging to tensorboard.
class LayerActivationMonitoring(BaseCallback):
        
    def _on_rollout_start(self) -> None:
        """Called after the training phase."""
        
        hooks = self.model.policy.features_extractor.hooks
        
        # Remove the hooks so that they don't get called for rollout collection.
        for h in hooks: h.remove() 

        # Log last datapoint and statistics to tensorboard.
        for i, hook in enumerate(hooks):
            if len(hook.activation_data) > 0:
                data = hook.activation_data[-1]
                self.logger.record(f'diagnostics/activation_l{i}', data)
                self.logger.record(f'diagnostics/mean_l{i}', np.mean(data))
                self.logger.record(f'diagnostics/std_l{i}', np.std(data))
                self.logger.record(f'diagnostics/low_act_prop_l{i}', get_low_act(data))

    def _on_rollout_end(self) -> None:
        """Called before the training phase."""
        for h in self.model.policy.features_extractor.hooks: h.register()

    def _on_step(self):
        pass
