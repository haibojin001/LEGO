# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg647::torch.cat+torch.no_grad
# name: torch_primitive
# summary: Uses torch.cat, torch.no_grad across 2 repos
# anchor_symbols: ['torch.cat', 'torch.no_grad']
# observed in 2 repos: ['pykale__pykale', 'zama-ai__concrete-ml']...

# --- from zama-ai__concrete-ml::use_case_examples/resnet/run_resnet18_fhe.py::evaluate_model ---
def evaluate_model(model, processor):
    with torch.no_grad():
        device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        outputs = []
        all_labels = []
        for batch in processor.dataloader:
            batch_images = batch["pixel_values"].to(device)
            batch_outputs = model(batch_images)
            outputs.append(batch_outputs)
            all_labels.append(batch["labels"])

        outputs = torch.cat(outputs)
        outputs = outputs.cpu()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu()

    accuracy = processor.accuracy(outputs, all_labels)
    topk_accuracy = processor.accuracy_top5(outputs, all_labels)
    print(f"Accuracy of the ResNet18 model on the images: {accuracy:.2f}%")
    print(f"Top-5 Accuracy of the ResNet18 model on the images: {topk_accuracy:.2f}%")

# --- from pykale__pykale::kale/pipeline/video_domain_adapter.py::WDGRLTrainerVideo.critic_update_steps ---
def critic_update_steps(self, batch):
        if self.current_epoch < self._init_epochs:
            return

        set_requires_grad(self.domain_classifier, requires_grad=True)

        if self.image_modality in ["rgb", "flow"]:
            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.rgb_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.rgb_feat(x_tu).data.view(x_tu.shape[0], -1)
            else:
                set_requires_grad(self.flow_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.flow_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.flow_feat(x_tu).data.view(x_tu.shape[0], -1)

            for _ in range(self._k_critic):
                # gp refers to gradient penelty in Wasserstein distance.
                gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

                critic_s = self.domain_classifier(h_s)
                critic_t = self.domain_classifier(h_t)
                wasserstein_distance = critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean()

                critic_cost = -wasserstein_distance + self._gamma * gp

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=True)
            else:
                set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)

        elif self.image_modality == "joint":
            set_requires_grad(self.rgb_feat, requires_grad=False)
            set_requires_grad(self.flow_feat, requires_grad=False)
            (x_s_rgb, y_s), (x_s_flow, _), (x_tu_rgb, _), (x_tu_flow, _) = batch
            with torch.no_grad():
                h_s_rgb = self.rgb_feat(x_s_rgb).data.view(x_s_rgb.shape[0], -1)
                h_t_rgb = self.rgb_feat(x_tu_rgb).data.view(x_tu_rgb.shape[0], -1)
                h_s_flow = self.flow_feat(x_s_flow).data.view(x_s_flow.shape[0], -1)
                h_t_flow = self.flow_feat(x_tu_flow).data.view(x_tu_flow.shape[0], -1)
                h_s = torch.cat((h_s_rgb, h_s_flow), dim=1)
                h_t = torch.cat((h_t_rgb, h_t_flow), dim=1)

            # Need to improve to process rgb and flow separately in the future.
            for _ in range(self._k_critic):
                # gp_x refers to gradient penelty for the input with the modality x.
                gp_rgb = losses.gradient_penalty(self.domain_classifier, h_s_rgb, h_t_rgb)
                gp_flow = losses.gradient_penalty(self.domain_classifier, h_s_flow, h_t_flow)

                critic_s_rgb = self.domain_classifier(h_s_rgb)
                critic_s_flow = self.domain_classifier(h_s_flow)
                critic_t_rgb = self.domain_classifier(h_t_rgb)
                critic_t_flow = self.domain_classifier(h_t_flow)
                wasserstein_distance_rgb = critic_s_rgb.mean() - (1 + self._beta_ratio) * critic_t_rgb.mean()
                wasserstein_distance_flow = critic_s_flow.mean() - (1 + self._beta_ratio) * critic_t_flow.mean()

                critic_cost = (
                    -wasserstein_distance_rgb
                    + -wasserstein_distance_flow
                    + self._gamma * gp_rgb
                    + self._gamma * gp_flow
                ) * 0.5

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            set_requires_grad(self.rgb_feat, requires_grad=True)
            set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)
