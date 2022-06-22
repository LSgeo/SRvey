print(f"Loading {__name__}")


def main():
    import srvey.cfg as cfg
    import srvey.data as data
    from srvey import Session
    from srvey.networks.lte import LTE

    session = Session(debug=True)

    train_dataloader, val_dataloader, preview_dataloader = data.build_dataloaders()
    # model = net.ArbRDNPlus(session, len(train_dataloader) * cfg.num_epochs)
    # model = net.RDNPlus(session, len(train_dataloader) * cfg.num_epochs)  # Locked 4x sr
    model = LTE(session)

    if cfg.pretrained_model_id:
        model.load_pretrained_model(cfg.pretrained_model)
        # Validate first if loading pre-trained model TODO

    model = model.to(session.device)
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class
    # or ?
    # model.to(session.device)
    # # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html#save-on-cpu-load-on-gpu

    for epoch_num in range(model.start_epoch, model.num_epochs):
        session.begin_epoch(epoch_num)
        model.curr_epoch += 1

        for iter_num, batch in enumerate(train_dataloader):
            model.curr_iteration += 1

            model.norm_feed_data(batch)
            model.train_on_batch()

            if iter_num % cfg.metric_freq == 0 and iter_num != 0:
                model.log_metrics()

        if epoch_num % cfg.val_freq == 0:
            for batch in val_dataloader:
                model.norm_feed_data(batch)
                model.validate_on_batch()
                model.log_metrics()

        if epoch_num % cfg.preview_freq == 0:
            for batch in preview_dataloader:
                model.norm_feed_data(batch)
                model.save_previews()

        if epoch_num % cfg.checkpoint_freq == 0:
            model.save_model(for_inference_only=False)

        if "mslr" in cfg.scheduler_spec["name"]:
            model.scheduler.step()

    model.save_model(name=f"final_epoch_{epoch_num}.tar", for_inference_only=False)
    session.end()


if __name__ == "__main__":
    main()
