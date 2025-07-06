import configparser
import os


class ConfigParser:
    """Configuration file parser"""
    
    def __init__(self, path=None):
        self.config = configparser.ConfigParser()
        self.config_path = path or "config.ini"
        
        # Create config file if it doesn't exist
        if not os.path.exists(self.config_path):
            self.create_default_config()
        else:
            self.config.read(self.config_path)
    
    def create_default_config(self):
        """Create default configuration file"""
        self.config['DEFAULT'] = {
            'IP': '192.168.1.99',
            'USER': 'admin',
            'PASSWORD': 'Flex'
        }
        self.save_config()
    
    def get_val(self, section, key, default=None):
        """Get configuration value"""
        try:
            return self.config.get(section, key)
        except:
            return default
    
    def set_val(self, section, key, value):
        """Set configuration value"""
        if section not in self.config:
            self.config.add_section(section)
        self.config.set(section, key, str(value))
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)