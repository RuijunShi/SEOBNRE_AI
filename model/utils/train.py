def train(model, batch, device, optimizer):
    par, target_waveforms_amp, target_waveforms_pha, mask = batch
    par = par.to(device).float()
    target_waveforms = target_waveforms_amp.to(device).float()
    optimizer.zero_grad()
    mask = mask.to(device).float()
    model(par, target_waveforms)
    loss_total = model.total_loss
    loss_total.backward()
    optimizer.step()
