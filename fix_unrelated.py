# Read current file
with open('unrelated.py', 'r') as f:
    lines = f.readlines()

# Find and replace sections to make plots more formal
output = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check for the figure command at line with MAIN FIT PLOT
    if 'plt.figure(figsize=(8,6))' in line and i > 100:  # Ensure we're in the right section
        # This is the main fit plot - replace with formal version
        output.append('plt.figure(figsize=(10,7), dpi=150)\n')
        i += 1
        # Now handle the errorbar call
        while i < len(lines) and 'plt.plot' not in lines[i]:
            if 'label="Data"' in lines[i]:
                output.append('color=\'black\',\n')
                output.append('label="Experimental Data"\n')
            elif 'capsize=4,' in lines[i]:
                output.append('markersize=8,\n')
                output.append('capsize=5,\n')
                output.append('elinewidth=1.5,\n')
            elif 'fmt=' in lines[i] or 'xerr=' in lines[i] or 'yerr=' in lines[i]:
                output.append(lines[i])
            i += 1
        i -= 1  # Back up one since the loop went one too far
    
    elif 'plt.plot(f_line, V_line, color=' in line and 'Best Fit' in line:
        output.append('plt.plot(f_line, V_line, color=\'red\', linewidth=2.5, label=\'Linear Fit\')\n')
        i += 1
    
    elif 'eq_text = f"V = ({m:.4f})f + ({b:.3f})"' in line:
        output.append('eq_text = f"$V = ({m:.4f})f + ({b:.3f})$"\n')
        i += 1
    
    elif 'fontsize=12,' in line and i > 100:
        output.append('fontsize=13,\n')
        i += 1
    
    elif 'verticalalignment=\'top\'' in line and i > 100 and 'MAIN' in ''.join(lines[max(0,i-20):i]):
        output.append('verticalalignment=\'top\',\n')
        i += 1
        # Update bbox
        output.append('bbox=dict(boxstyle=\'round\', facecolor=\'white\', alpha=0.95, edgecolor=\'black\', linewidth=1)\n')
        # Skip the old bbox line
        while i < len(lines) and 'bbox=' not in lines[i]:
            i += 1
        i += 1  # Skip the bbox line
    
    elif 'plt.xlabel("Frequency (THz)")' in line and i > 100:
        output.append('plt.xlabel("Frequency (THz)", fontsize=14, fontweight=\'bold\')\n')
        i += 1
    
    elif 'plt.ylabel("Stopping Voltage (V)")' in line and i > 100:
        output.append('plt.ylabel("Stopping Voltage (V)", fontsize=14, fontweight=\'bold\')\n')
        i += 1
    
    elif 'plt.title("Stopping Voltage vs Frequency")' in line:
        output.append('plt.title("Stopping Voltage vs Frequency", fontsize=15, fontweight=\'bold\', pad=20)\n')
        i += 1
    
    elif 'plt.legend()' in line and 'MAIN' in ''.join(lines[max(0,i-20):i]):
        output.append('plt.legend(fontsize=12, loc=\'lower right\')\n')
        i += 1
    
    elif 'plt.grid()' in line and 'MAIN' in ''.join(lines[max(0,i-20):i]) and i > 100:
        output.append('plt.grid(True, alpha=0.3, linestyle=\'--\', linewidth=0.5)\n')
        output.append('plt.tight_layout()\n')
        i += 1
    
    # Handle RESIDUALS PLOT
    elif 'plt.figure(figsize=(8,4))' in line:
        output.append('plt.figure(figsize=(10,6), dpi=150)\n')
        i += 1
    
    elif 'plt.axhline(0, linestyle=\'--\')' in line:
        output.append('plt.axhline(0, linestyle=\'-\', color=\'red\', linewidth=2.5, alpha=0.7)\n')
        i += 1
    
    elif 'plt.xlabel("Frequency (THz)")' in line and 'RESIDUALS' in ''.join(lines[max(0,i-10):i]):
        output.append('plt.xlabel("Frequency (THz)", fontsize=14, fontweight=\'bold\')\n')
        i += 1
    
    elif 'plt.ylabel("Residuals (V)")' in line:
        output.append('plt.ylabel("Residuals (V)", fontsize=14, fontweight=\'bold\')\n')
        i += 1
    
    elif 'plt.title("Residuals of Linear Fit")' in line:
        output.append('plt.title("Residuals of Linear Fit", fontsize=15, fontweight=\'bold\', pad=20)\n')
        i += 1
    
    elif 'plt.grid()' in line and 'RESIDUALS' in ''.join(lines[max(0,i-20):i]):
        output.append('plt.grid(True, alpha=0.3, linestyle=\'--\', linewidth=0.5)\n')
        i += 1
    
    else:
        output.append(line)
        i += 1

# Write the modified content back
with open('unrelated.py', 'w') as f:
    f.writelines(output)

print("File has been updated successfully!")
