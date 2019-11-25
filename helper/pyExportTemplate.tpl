{%- extends 'python.tpl' -%}

{%- block input_group -%}

    {%- if 'tags' in cell.metadata -%}
    
        {%- if 'hide-cell' in cell.metadata['tags'] -%}
        
            {%- for x in super().split('\n') -%} 
                {{ "#" +  x + "\n" }}
            {%- endfor -%}
            
        {%- elif 'show-cell' in cell.metadata['tags'] -%}
        
            {%- for x in super().split('\n')[2:] -%}
                {{ x.split('#')[-1] + "\n" }}
                
            {%- endfor -%}
            
        {%- endif -%}
        
    {%- else -%}
        {{ super() }}
    
    
    {%- endif -%}

{%- endblock input_group -%}
