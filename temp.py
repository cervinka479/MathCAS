def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, task='class'):
    # Initialize lists to store losses and validation accuracies
    train_losses = []
    val_losses = []
    epoch_times = []
    val_accuracies = [] if task == 'class' else None
    
    # Variable to store the best validation loss
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    
    # Training loop
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
    
        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                
                if task == 'class':
                    # Calculate accuracy
                    predicted = (outputs > 0.5).float()
                    correct_val += (predicted == batch_y).sum().item()
                    total_val += batch_y.size(0)
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        if task == 'class':
            val_accuracies.append(correct_val / total_val)

        # Save the model if it has the best validation loss so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'shear_50M.pth')
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # Calculate elapsed time and estimated time left
        elapsed_time = time.time() - start_time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(elapsed_time)  # Record the total elapsed time for this epoch
        estimated_time_left = epoch_time * (num_epochs - epoch - 1)
        
        # Format time in hours:minutes:seconds
        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
        estimated_time_left_str = str(timedelta(seconds=int(estimated_time_left)))
        
        # Print epoch information
        if task == 'class':
            print(f"Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Accuracy: {correct_val / total_val:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        else:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Val loss: {epoch_val_loss:.4f}, Elapsed time: {elapsed_time_str}, Estimated time left: {estimated_time_left_str}")
        sys.stdout.flush()

        # Save losses and validation accuracies to CSV file
        df_metrics = pd.DataFrame({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'elapsed_time': epoch_times,
        })
        if task == 'class':
            df_metrics['val_accuracies'] = val_accuracies
        df_metrics.to_csv(f'shear_50M_metrics.csv', index=False)

    # Print a newline after the final epoch information
    print()
