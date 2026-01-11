#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_report.py

Script para juntar todos os arquivos markdown do relatório técnico
em um único arquivo consolidado.

Uso:
    python merge_report.py
    
Saída:
    RELATORIO_TECNICO_COMPLETO.md
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Ordem dos arquivos para merge
CHAPTER_FILES = [
    "cap1_parte1_stack_tecnologico.md",
    "cap1_parte2_superbuild_ndk.md",
    "cap2_parte1_sfm_pipeline.md",
    "cap2_parte2_bundle_adjustment_fast.md",
    "cap3_parte1_memoria_pointcloud.md",
    "cap3_parte2_tiling_cog.md",
    "cap4_parte1_matematica_camera.md",
    "cap4_parte2_georef_indices.md",
    "cap5_parte1_arquitetura_jni.md",
    "cap5_parte2_implementacao_pipeline.md",
]

# Cabeçalho do relatório final
REPORT_HEADER = """
================================================================================
     ANÁLISE TÉCNICA EXAUSTIVA DO CÓDIGO-FONTE WebODM/ODM
     BLUEPRINT DE IMPLEMENTAÇÃO: ANDROID FAST STITCHING (TABLET OFFLINE)
================================================================================

                          RELATÓRIO TÉCNICO COMPLETO
                          
Gerado em: {timestamp}
Versão: 1.0

================================================================================
                                ÍNDICE
================================================================================

CAPÍTULO 1: STACK TECNOLÓGICO E PORTABILIDADE
  1.1 Arquitetura Geral do WebODM
  1.2 Linguagens de Programação
  1.3 Framework Web (Django)
  1.4 Pipeline de Processamento
  1.5 Dependências Python
  1.6 Dependências JavaScript/Node.js
  1.7 Docker e Containerização
  1.8 SuperBuild - Sistema de Compilação
  1.9 Compilação para Android (NDK)
  1.10 Conclusões do Capítulo 1

CAPÍTULO 2: MOTOR SfM E MODO FAST
  2.1 Pipeline de Processamento Completo
  2.2 OpenSfM - Arquitetura
  2.3 Extração de Features
  2.4 Matching de Features
  2.5 Reconstrução Incremental
  2.6 Bundle Adjustment
  2.7 Modo Fast Orthophoto
  2.8 Parâmetros Críticos
  2.9 Conclusões do Capítulo 2

CAPÍTULO 3: GESTÃO DE MEMÓRIA E TILING
  3.1 Estratégias de Memória do ODM
  3.2 PDAL e Processamento de Point Clouds
  3.3 Split-Merge para Datasets Grandes
  3.4 Classificação de Ground Points
  3.5 Sistema de Tiling
  3.6 Cloud Optimized GeoTIFF (COG)
  3.7 Entwine Point Tiles (EPT)
  3.8 Estratégias para Android
  3.9 Conclusões do Capítulo 3

CAPÍTULO 4: FUNDAMENTOS MATEMÁTICOS DE AGRIMENSURA
  4.1 Modelo de Câmera Pinhole
  4.2 Matriz Intrínseca
  4.3 Modelos de Distorção
  4.4 Geometria Epipolar
  4.5 Triangulação
  4.6 Problema PnP
  4.7 Sistemas de Coordenadas
  4.8 Georreferenciamento
  4.9 Geração de DEM
  4.10 Índices Vegetativos
  4.11 Fórmulas de Área e Volume
  4.12 Conclusões do Capítulo 4

CAPÍTULO 5: BLUEPRINT DE IMPLEMENTAÇÃO ANDROID
  5.1 Visão Geral da Arquitetura
  5.2 Interface JNI
  5.3 Compilação Nativa com NDK
  5.4 Gestão de Memória no Android
  5.5 Pipeline de Processamento Android
  5.6 UseCase de Processamento
  5.7 Interface de Usuário (Jetpack Compose)
  5.8 Exportação de Resultados
  5.9 Otimizações para Tablet
  5.10 Diagrama de Sequência
  5.11 Conclusões e Roadmap

================================================================================

"""

REPORT_FOOTER = """

================================================================================
                              FIM DO RELATÓRIO
================================================================================

Este documento foi gerado automaticamente a partir dos arquivos de análise
técnica do projeto WebODM.

Para mais informações, consulte:
- Repositório WebODM: https://github.com/OpenDroneMap/WebODM
- Repositório ODM: https://github.com/OpenDroneMap/ODM
- Documentação: https://docs.opendronemap.org/

================================================================================
                    © {year} - Análise Técnica Fast Stitching
================================================================================
"""


def find_script_directory() -> Path:
    """Encontrar o diretório onde o script está localizado."""
    return Path(__file__).parent.resolve()


def read_file_content(filepath: Path) -> str:
    """Ler conteúdo de um arquivo markdown."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"⚠️  Arquivo não encontrado: {filepath}")
        return f"\n\n[ARQUIVO NÃO ENCONTRADO: {filepath.name}]\n\n"
    except Exception as e:
        print(f"❌ Erro ao ler {filepath}: {e}")
        return f"\n\n[ERRO AO LER: {filepath.name}]\n\n"


def add_page_break() -> str:
    """Adicionar separador visual entre capítulos."""
    return "\n\n" + "=" * 80 + "\n\n"


def merge_reports(source_dir: Path, output_file: Path) -> bool:
    """
    Juntar todos os arquivos markdown em um único relatório.
    
    Args:
        source_dir: Diretório contendo os arquivos de capítulo
        output_file: Caminho do arquivo de saída
        
    Returns:
        True se sucesso, False caso contrário
    """
    print("=" * 60)
    print("  MERGE DE RELATÓRIO TÉCNICO")
    print("=" * 60)
    print()
    
    # Verificar diretório fonte
    if not source_dir.exists():
        print(f"❌ Diretório não encontrado: {source_dir}")
        return False
    
    print(f"📁 Diretório fonte: {source_dir}")
    print(f"📄 Arquivo de saída: {output_file}")
    print()
    
    # Preparar conteúdo
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year = datetime.now().year
    
    content_parts = []
    
    # Adicionar cabeçalho
    content_parts.append(REPORT_HEADER.format(timestamp=timestamp))
    
    # Processar cada arquivo de capítulo
    successful = 0
    failed = 0
    
    for i, filename in enumerate(CHAPTER_FILES, 1):
        filepath = source_dir / filename
        print(f"[{i:2d}/{len(CHAPTER_FILES)}] Processando: {filename}...", end=" ")
        
        if filepath.exists():
            content = read_file_content(filepath)
            content_parts.append(content)
            content_parts.append(add_page_break())
            print("✅")
            successful += 1
        else:
            content_parts.append(f"\n\n[ARQUIVO PENDENTE: {filename}]\n\n")
            content_parts.append(add_page_break())
            print("⚠️  NÃO ENCONTRADO")
            failed += 1
    
    # Adicionar rodapé
    content_parts.append(REPORT_FOOTER.format(year=year))
    
    # Juntar tudo
    final_content = "".join(content_parts)
    
    # Escrever arquivo de saída
    print()
    print("📝 Escrevendo arquivo final...", end=" ")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print("✅")
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False
    
    # Estatísticas finais
    print()
    print("=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  ✅ Arquivos processados com sucesso: {successful}")
    print(f"  ⚠️  Arquivos não encontrados: {failed}")
    print(f"  📄 Tamanho do relatório final: {len(final_content):,} caracteres")
    print(f"  📄 Linhas totais: {final_content.count(chr(10)):,}")
    print()
    print(f"  ✅ Relatório salvo em: {output_file}")
    print("=" * 60)
    
    return True


def main():
    """Função principal."""
    # Determinar diretórios
    script_dir = find_script_directory()
    source_dir = script_dir  # Os arquivos .md estão no mesmo diretório
    output_file = script_dir / "RELATORIO_TECNICO_COMPLETO.md"
    
    # Executar merge
    success = merge_reports(source_dir, output_file)
    
    # Código de saída
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
