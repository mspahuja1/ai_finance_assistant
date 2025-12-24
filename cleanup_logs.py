"""Cleanup script for log files"""
import os
import shutil
from datetime import datetime
import argparse

# Define log directory
LOG_DIR = "logs"

def get_log_files(log_dir):
    """Get all log files in the directory"""
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist")
        return []
    
    log_files = []
    for file in os.listdir(log_dir):
        if file.endswith('.log'):
            filepath = os.path.join(log_dir, file)
            log_files.append(filepath)
    
    return log_files

def get_file_size(filepath):
    """Get file size in KB"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / 1024

def backup_logs(log_dir, backup_dir=None):
    """Backup logs before cleaning"""
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"logs_backup_{timestamp}"
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist")
        return False
    
    try:
        shutil.copytree(log_dir, backup_dir)
        print(f"✅ Logs backed up to: {backup_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to backup logs: {e}")
        return False

def clean_logs(log_dir, backup=True):
    """Clean all log files"""
    print("=" * 70)
    print("🧹 LOG CLEANUP UTILITY")
    print("=" * 70)
    
    log_files = get_log_files(log_dir)
    
    if not log_files:
        print("ℹ️  No log files found to clean")
        return
    
    # Show current log files
    print(f"\n📂 Found {len(log_files)} log file(s):")
    total_size = 0
    for filepath in log_files:
        size_kb = get_file_size(filepath)
        total_size += size_kb
        print(f"   - {os.path.basename(filepath)}: {size_kb:.2f} KB")
    
    print(f"\n📊 Total size: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    
    # Backup if requested
    if backup:
        print("\n🔄 Creating backup...")
        if not backup_logs(log_dir):
            response = input("\n⚠️  Backup failed. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Cleanup cancelled")
                return
    
    # Confirm deletion
    print("\n⚠️  This will DELETE all log files!")
    response = input("Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    # Delete log files
    print("\n🗑️  Deleting log files...")
    deleted_count = 0
    failed_count = 0
    
    for filepath in log_files:
        try:
            os.remove(filepath)
            print(f"   ✅ Deleted: {os.path.basename(filepath)}")
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ Failed to delete {os.path.basename(filepath)}: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 CLEANUP SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully deleted: {deleted_count} file(s)")
    if failed_count > 0:
        print(f"❌ Failed to delete: {failed_count} file(s)")
    print(f"💾 Space freed: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    print("=" * 70)

def truncate_logs(log_dir):
    """Truncate log files (empty them but keep files)"""
    print("=" * 70)
    print("📝 LOG TRUNCATE UTILITY")
    print("=" * 70)
    
    log_files = get_log_files(log_dir)
    
    if not log_files:
        print("ℹ️  No log files found to truncate")
        return
    
    print(f"\n📂 Found {len(log_files)} log file(s):")
    total_size = 0
    for filepath in log_files:
        size_kb = get_file_size(filepath)
        total_size += size_kb
        print(f"   - {os.path.basename(filepath)}: {size_kb:.2f} KB")
    
    print(f"\n📊 Total size: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    print("\n⚠️  This will EMPTY all log files (but keep the files)!")
    response = input("Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("❌ Truncate cancelled")
        return
    
    print("\n📝 Truncating log files...")
    truncated_count = 0
    failed_count = 0
    
    for filepath in log_files:
        try:
            with open(filepath, 'w') as f:
                f.write('')
            print(f"   ✅ Truncated: {os.path.basename(filepath)}")
            truncated_count += 1
        except Exception as e:
            print(f"   ❌ Failed to truncate {os.path.basename(filepath)}: {e}")
            failed_count += 1
    
    print("\n" + "=" * 70)
    print("📊 TRUNCATE SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully truncated: {truncated_count} file(s)")
    if failed_count > 0:
        print(f"❌ Failed to truncate: {failed_count} file(s)")
    print(f"💾 Space freed: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    print("=" * 70)

def archive_logs(log_dir):
    """Archive logs to a zip file"""
    print("=" * 70)
    print("📦 LOG ARCHIVE UTILITY")
    print("=" * 70)
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"logs_archive_{timestamp}"
    
    try:
        print(f"\n📦 Creating archive: {archive_name}.zip")
        shutil.make_archive(archive_name, 'zip', log_dir)
        archive_size = os.path.getsize(f"{archive_name}.zip") / 1024
        print(f"✅ Archive created: {archive_name}.zip ({archive_size:.2f} KB)")
        
        response = input("\n🗑️  Delete original log files after archiving? (y/N): ")
        if response.lower() == 'y':
            clean_logs(log_dir, backup=False)
    except Exception as e:
        print(f"❌ Failed to create archive: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cleanup utility for log files')
    parser.add_argument(
        'action',
        choices=['clean', 'truncate', 'archive', 'backup', 'list'],
        help='Action to perform: clean (delete), truncate (empty), archive (zip), backup, or list'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup before cleaning (use with caution!)'
    )
    parser.add_argument(
        '--log-dir',
        default='logs',
        help='Log directory path (default: logs)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'list':
        log_files = get_log_files(args.log_dir)
        if not log_files:
            print("ℹ️  No log files found")
            return
        
        print(f"\n📂 Log files in '{args.log_dir}':")
        total_size = 0
        for filepath in log_files:
            size_kb = get_file_size(filepath)
            total_size += size_kb
            print(f"   - {os.path.basename(filepath)}: {size_kb:.2f} KB")
        print(f"\n📊 Total: {len(log_files)} file(s), {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    
    elif args.action == 'clean':
        clean_logs(args.log_dir, backup=not args.no_backup)
    
    elif args.action == 'truncate':
        truncate_logs(args.log_dir)
    
    elif args.action == 'archive':
        archive_logs(args.log_dir)
    
    elif args.action == 'backup':
        backup_logs(args.log_dir)

if __name__ == "__main__":
    main()